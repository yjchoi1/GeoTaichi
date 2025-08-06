import taichi as ti

from src.utils.constants import DBL_EPSILON
from src.utils.TypeDefination import mat2x2, vec2f
from src.utils.MatrixFunction import inverse_matrix_2x2
from src.utils.VectorFunction import Squared
from src.utils.ScalarFunction import clamp

TOL = 100 * DBL_EPSILON


@ti.data_oriented
class NurbsBasisFunction:
    def __init__(self, degree):
        self.degree = degree

    @ti.func
    def FindSpanLinear(self, start_knot, num_ctrlpts, knot, knot_vector):
        span = self.degree + 1  
        while span < num_ctrlpts and knot_vector[span + start_knot] <= knot:
            span += 1
        return span - 1

    @ti.func
    def FindSpan(self, start_knot, num_ctrlpts, knot, knot_vector):
        low = self.degree
        high = num_ctrlpts
        span = ti.cast(0.5 * (low + high), int)
        if knot > knot_vector[num_ctrlpts + start_knot]:
            span = num_ctrlpts - 1
        elif knot < knot_vector[self.degree + start_knot]:
            span = self.degree
        else:
            if knot == knot_vector[num_ctrlpts + start_knot]: 
                span = num_ctrlpts - 1
            else:
                while (knot < knot_vector[span + start_knot] or knot >= knot_vector[span + 1 + start_knot]) and high != low:
                    if knot < knot_vector[span + start_knot]: 
                        high = span
                    else:                
                        low = span
                    span = ti.cast(0.5 * (low + high), int)
        return span
        
    @ti.func
    def BasisFuncs(self, start_knot, span, knot, knot_vector):
        Nurbs = ti.Vector.zero(float, self.degree + 1)
        left = ti.Vector.zero(float, self.degree + 1)
        right = ti.Vector.zero(float, self.degree + 1)
        Nurbs[0] = 1.0

        for i in ti.static(range(1, self.degree + 1)):
            left[i] = knot - knot_vector[span + 1 - i + start_knot]
            right[i] = knot_vector[span + i + start_knot] - knot
            saved = 0.

            for j in range(i):
                temp = Nurbs[j] / (right[j + 1] + left[i - j])
                Nurbs[j] = saved + right[j + 1] * temp
                saved = left[i - j] * temp
            Nurbs[i] = saved
        return Nurbs

    @ti.func
    def FirstDersBasisFuncs(self, start_knot, span, knot, knot_vector):
        derNurbs = ti.Matrix.zero(float, 2, self.degree + 1)
        left = ti.Vector.zero(float, self.degree + 1)
        right = ti.Vector.zero(float, self.degree + 1)
        ndu = ti.Matrix.zero(float, self.degree + 1, self.degree + 1)
        a = ti.Matrix.zero(float, 2, self.degree + 1)

        ndu[0, 0] = 1.
        for i in ti.static(range(1, self.degree + 1)):
            left[i] = knot - knot_vector[span + 1 - i + start_knot]
            right[i] = knot_vector[span + i + start_knot] - knot
            saved = 0.
            for j in range(i):
                ndu[i, j] = right[j + 1] + left[i - j]
                temp = ndu[j, i - 1] / ndu[i, j]

                ndu[j, i] = saved + right[j + 1] * temp
                saved = left[i - j] * temp
            ndu[i, i] = saved

        for i in ti.static(range(self.degree + 1)):
            derNurbs[0, i] = ndu[i, self.degree]

        for r in ti.static(range(self.degree + 1)):
            s1 = 0
            s2 = 1
            a[0, 0] = 1.
            k = 1

            # the first derviative
            d = 0.
            rk = r - k
            pk = self.degree - k
            if r >= k:
                a[s2, 0] = a[s1, 0] / ndu[pk + 1, rk]
                d = a[s2, 0] * ndu[rk, pk]
            j1 = 1 if rk >= -1 else -rk
            j2 = k - 1 if r - 1 <= pk else self.degree - r
            for j in range(j1, j2 + 1):
                a[s2, j] = (a[s1, j] - a[s1, j - 1]) / ndu[pk + 1, rk + j]
                d += a[s2, j] * ndu[rk + j, pk]
            
            if r <= pk:
                a[s2, k] = -a[s1, k - 1] / ndu[pk + 1, r]
                d += a[s2, k] * ndu[r, pk]
            derNurbs[k, r] = d
            s1, s2 = s2, s1
        
        r = self.degree
        for j in ti.static(range(self.degree + 1)):
            derNurbs[1, j] *= r
        r *= (self.degree - 1)
        return derNurbs

    @ti.func
    def SecondDersBasisFuncs(self, start_knot, span, knot, knot_vector):
        derNurbs = ti.Matrix.zero(float, 3, self.degree + 1)
        left = ti.Vector.zero(float, self.degree + 1)
        right = ti.Vector.zero(float, self.degree + 1)
        ndu = ti.Matrix.zero(float, self.degree + 1, self.degree + 1)
        a = ti.Matrix.zero(float, 2, self.degree + 1)

        ndu[0, 0] = 1.
        for i in ti.static(range(1, self.degree + 1)):
            left[i] = knot - knot_vector[span + 1 - i + start_knot]
            right[i] = knot_vector[span + i + start_knot] - knot
            saved = 0.
            for j in range(i):
                ndu[i, j] = right[j + 1] + left[i - j]
                temp = ndu[j, i - 1] / ndu[i, j]

                ndu[j, i] = saved + right[j + 1] * temp
                saved = left[i - j] * temp
            ndu[i, i] = saved

        for i in ti.static(range(self.degree + 1)):
            derNurbs[0, i] = ndu[i, self.degree]

        for r in ti.static(range(self.degree + 1)):
            s1 = 0
            s2 = 1
            a[0, 0] = 1.

            for k in ti.static(range(1, 3)):
                d = 0.
                rk = r - k
                pk = self.degree - k
                if r >= k:
                    a[s2, 0] = a[s1, 0] / ndu[pk + 1, rk]
                    d = a[s2, 0] * ndu[rk, pk]
                j1 = 1 if rk >= -1 else -rk
                j2 = k - 1 if r - 1 <= pk else self.degree - r
                for j in range(j1, j2 + 1):
                    a[s2, j] = (a[s1, j] - a[s1, j - 1]) / ndu[pk + 1, rk + j]
                    d += a[s2, j] * ndu[rk + j, pk]
                
                if r <= pk:
                    a[s2, k] = -a[s1, k - 1] / ndu[pk + 1, r]
                    d += a[s2, k] * ndu[r, pk]
                derNurbs[k, r] = d
                s1, s2 = s2, s1
        
        r = self.degree
        for k in ti.static(range(1, 3)):
            for j in ti.static(range(self.degree + 1)):
                derNurbs[k, j] *= r
            r *= (self.degree - k)
        return derNurbs
    

@ti.data_oriented
class NurbsBasisFunction1d:
    def __init__(self, degree, dimension=3):
        self.basis_u = NurbsBasisFunction(degree)
        self.dimension = dimension

    @ti.func
    def FindElement(self, start_knot, num_knot, xi, knot_vector):
        num_ctrlpts_u = num_knot - self.basis_u.degree - 1
        spanU = self.basis_u.FindSpan(start_knot, num_ctrlpts_u, xi, knot_vector)
        return spanU - self.basis_u.degree

    @ti.func
    def NurbsBasis1d(self, start_knot, start_ctrlpt, num_knot, xi, knot_vector, weight):
        N = ti.Vector.zero(float, (self.basis_u.degree + 1))
        num_ctrlpts_u = num_knot - self.basis_u.degree - 1

        spanU = self.basis_u.FindSpan(start_knot, num_ctrlpts_u, xi, knot_vector)
        NurbsU = self.basis_u.BasisFuncs(start_knot, spanU, xi, knot_vector)

        w, kk = 0., 0
        uind = spanU - self.basis_u.degree
        for k in ti.static(range(self.basis_u.degree + 1)):
            linear_id = uind + k + start_ctrlpt
            nmp = NurbsU[k] 
            w += nmp * weight[linear_id]
            N[kk] = nmp * weight[linear_id]
            kk += 1
        return N / w
    
    @ti.func
    def NurbsBasisDers1d(self, start_knot, start_ctrlpt, num_knot, xi, knot_vector, weight):
        N = ti.Vector.zero(float, (self.basis_u.degree + 1))
        dNdnat = ti.Vector.zero(float, (self.basis_u.degree + 1))
        num_ctrlpts = num_knot - 1 - self.basis_u.degree 

        spanU = self.basis_u.FindSpan(start_knot, num_ctrlpts, xi, knot_vector)
        dNurbs = self.basis_u.FirstDersBasisFuncs(start_knot, spanU, xi, knot_vector)

        w, dwdxi = 0., 0.
        uind = spanU - self.basis_u.degree
        for k in ti.static(range(self.basis_u.degree + 1)):
            linear_id = uind + k + start_ctrlpt
            wgt = weight[linear_id]
            w += dNurbs[0, k] * wgt
            dwdxi += dNurbs[1, k] * wgt

        kk = 0
        uind = spanU - self.basis_u.degree
        for k in ti.static(range(self.basis_u.degree + 1)):
            linear_id = uind + k + start_ctrlpt
            fac = weight[linear_id] / (w * w)
            nmp = dNurbs[0, k]
            N[kk] = nmp * fac * w
            dNdnat[kk] = (dNurbs[1, k] * w - nmp * dwdxi) * fac
            kk += 1
        return N, dNdnat                    

    @ti.func
    def NurbsBasis2ndDers1d(self, start_knot, start_ctrlpt, num_knot, xi, knot_vector, weight):
        N = ti.Vector.zero(float, (self.basis_u.degree + 1))
        dNdnat = ti.Vector.zero(float, (self.basis_u.degree + 1))
        d2Ndnat = ti.Vector.zero(float, (self.basis_u.degree + 1))
        num_ctrlpts = num_knot - 1 - self.basis_u.degree

        span = self.basis_u.FindSpan(start_knot, num_ctrlpts, xi, knot_vector)
        Nurbs = self.basis_u.SecondDersBasisFuncs(start_knot, span, xi, knot_vector)

        w = 0.0
        dwdxi, d2wdxi = 0.0, 0.0
        uind = span - self.basis_u.degree
        for k in ti.static(range(self.basis_u.degree + 1)):
            linear_id = uind + k + start_ctrlpt
            wgt = weight[linear_id]
            w += Nurbs[0, k] * wgt
            dwdxi += Nurbs[1, k] * wgt
            d2wdxi += Nurbs[2, k] * wgt

        kk = 0
        inv_w = 1 / w
        inv_w2 = 1 / w / w
        inv_w3 = 1 / w / w / w
        uind = span - self.basis_u.degree
        for k in ti.static(range(self.basis_u.degree + 1)):
            linear_id = uind + k + start_ctrlpt
            wgt = weight[linear_id]
            fac = wgt / (w * w)
            nmp = Nurbs[0, k]

            N[kk] = nmp * fac * w
            dNdnat[kk] = (Nurbs[1, k] * w - nmp * dwdxi) * fac
            d2Ndnat[kk] = (Nurbs[2, k] * inv_w - 2 * Nurbs[1, k] * dwdxi * inv_w2 - Nurbs[0, k] * d2wdxi * inv_w2 + 2 * Nurbs[0, k] * dwdxi * dwdxi * inv_w3) * wgt
        return N, dNdnat, d2Ndnat

    @ti.func
    def NurbsBasisInterpolations1d(self, start_knot, start_ctrlpt, num_knot, xi, knot_vector, point, weight):
        num_ctrlpts = num_knot - 1 - self.basis_u.degree

        span = self.basis_u.FindSpan(start_knot, num_ctrlpts, xi, knot_vector)
        Nurbs = self.basis_u.BasisFuncs(start_knot, span, xi, knot_vector)
        
        position, w = ti.Vector.zero(float, self.dimension), 0.0
        uind = span - self.basis_u.degree
        for k in ti.static(range(self.basis_u.degree + 1)):
            linear_id = uind + k + start_ctrlpt
            position += Nurbs[k] * weight[linear_id] * point[linear_id]
            w += Nurbs[k] * weight[linear_id]
        return position / w

    @ti.func
    def NurbsBasisInterpolationsDers1d(self, start_knot, start_ctrlpt, num_knot, xi, knot_vector, point, weight):
        num_ctrlpts = num_knot - 1 - self.basis_u.degree 

        spanU = self.basis_u.FindSpan(start_knot, num_ctrlpts, xi, knot_vector)
        dNurbs = self.basis_u.FirstDersBasisFuncs(start_knot, spanU, xi, knot_vector)

        w, dwdxi = 0., 0.
        uind = spanU - self.basis_u.degree 
        for k in ti.static(range(self.basis_u.degree + 1)):
            linear_id = uind + k + start_ctrlpt
            wgt = weight[linear_id]
            w += dNurbs[0, k] * wgt
            dwdxi += dNurbs[1, k] * wgt

        position, dirsU = ti.Vector.zero(float, self.dimension), ti.Vector.zero(float, self.dimension)
        uind = spanU - self.basis_u.degree 
        for k in ti.static(range(self.basis_u.degree + 1)):
            linear_id = uind + k + start_ctrlpt
            fac = weight[linear_id] / (w * w)
            nmp = dNurbs[0, k]
            position += nmp * fac * w * point[linear_id]
            dirsU += (dNurbs[1, k] * w - nmp * dwdxi) * fac * point[linear_id]
        return position, dirsU

    @ti.func
    def NurbsBasisInterpolations2ndDers1d(self, start_knot, start_ctrlpt, num_knot, xi, knot_vector, point, weight):
        num_ctrlpts = num_knot - 1 - self.basis_u.degree 

        span = self.basis_u.FindSpan(start_knot, num_ctrlpts, xi, knot_vector)
        dNurbs = self.basis_u.SecondDersBasisFuncs(start_knot, span, xi, knot_vector)

        w = 0.0
        dwdxi, d2wdxi = 0.0, 0.0
        uind = span - self.basis_u.degree
        for k in ti.static(range(self.basis_u.degree + 1)):
            linear_id = uind + k + start_ctrlpt
            wgt = weight[linear_id]
            w += dNurbs[0, k] * wgt
            dwdxi += dNurbs[1, k] * wgt
            d2wdxi += dNurbs[2, k] * wgt

        position, dirsU, ddirsUU = ti.Vector.zero(float, self.dimension), ti.Vector.zero(float, self.dimension), ti.Vector.zero(float, self.dimension)
        inv_w = 1 / w
        inv_w2 = 1 / w / w
        inv_w3 = 1 / w / w / w
        uind = span - self.basis_u.degree
        for k in ti.static(range(self.basis_u.degree + 1)):
            linear_id = uind + k + start_ctrlpt
            wgt = weight[linear_id]
            fac = wgt / (w * w)
            nmp = dNurbs[0, k]

            position += nmp * fac * w * point[linear_id]
            dirsU += (dNurbs[1, k] * w - nmp * dwdxi) * fac * point[linear_id]
            ddirsUU += (dNurbs[2, k] * inv_w - 2 * dNurbs[1, k] * dwdxi * inv_w2 - dNurbs[0, k] * d2wdxi * inv_w2 + 2 * dNurbs[0, k] * dwdxi * dwdxi * inv_w3) * wgt * point[linear_id]
        return position, dirsU, ddirsUU
    
    @ti.func
    def NurbsBasisHessian(self, start_knot, start_ctrlpt, num_knot, xi, knot_vector, point, weight):
        num_ctrlpts = num_knot - 1 - self.basis_u.degree 

        span = self.basis_u.FindSpan(start_knot, num_ctrlpts, xi, knot_vector)
        dNurbs = self.basis_u.SecondDersBasisFuncs(start_knot, span, xi, knot_vector)

        w = 0.0
        dwdxi, d2wdxi = 0.0, 0.0
        uind = span - self.basis_u.degree
        for k in ti.static(range(self.basis_u.degree + 1)):
            linear_id = uind + k + start_ctrlpt
            wgt = weight[linear_id]
            w += dNurbs[0, k] * wgt
            dwdxi += dNurbs[1, k] * wgt
            d2wdxi += dNurbs[2, k] * wgt

        position, dirsU, ddirsUU = ti.Vector.zero(float, self.dimension), ti.Vector.zero(float, self.dimension), ti.Vector.zero(float, self.dimension)
        N = ti.Vector.zero(float, (self.basis_u.degree + 1))
        dNdnat = ti.Vector.zero(float, (self.basis_u.degree + 1))
        d2Ndnat = ti.Vector.zero(float, (self.basis_u.degree + 1))
        inv_w = 1 / w
        inv_w2 = 1 / w / w
        inv_w3 = 1 / w / w / w
        kk = 0
        uind = span - self.basis_u.degree
        for k in ti.static(range(self.basis_u.degree + 1)):
            linear_id = uind + k + start_ctrlpt
            wgt = weight[linear_id]
            fac = wgt / (w * w)
            nmp = dNurbs[0, k]

            nshape = nmp * fac * w
            derivative = (dNurbs[1, k] * w - nmp * dwdxi) * fac
            dderivative = (dNurbs[2, k] * inv_w - 2 * dNurbs[1, k] * dwdxi * inv_w2 - dNurbs[0, k] * d2wdxi * inv_w2 + 2 * dNurbs[0, k] * dwdxi * dwdxi * inv_w3) * wgt

            position += nshape * point[linear_id]
            dirsU += derivative * point[linear_id]
            ddirsUU += dderivative * point[linear_id]
            
            N[kk] = nshape
            dNdnat[kk] = derivative
            d2Ndnat[kk] = dderivative
            kk += 1
        return span, N, dNdnat, dirsU, ddirsUU


@ti.data_oriented
class NurbsBasisFunction2d:
    def __init__(self, degree_u, degree_v, dimension=3):
        self.basis_u = NurbsBasisFunction(degree_u)
        self.basis_v = NurbsBasisFunction(degree_v)
        self.dimension = dimension

    @ti.func
    def FindElement(self, start_knot_u, start_knot_v, num_knot_u, num_knot_v, xi, eta, knot_vector_u, knot_vector_v):
        num_ctrlpts_u = num_knot_u - 1 - self.basis_u.degree
        num_ctrlpts_v = num_knot_v - 1 - self.basis_v.degree

        spanU = self.basis_u.FindSpan(start_knot_u, num_ctrlpts_u, xi, knot_vector_u)
        spanV = self.basis_v.FindSpan(start_knot_v, num_ctrlpts_v, eta, knot_vector_v)
        return spanU - self.basis_u.degree, spanV - self.basis_v.degree

    @ti.func
    def NurbsBasis2d(self, start_knot_u, start_knot_v, start_ctrlpt, num_knot_u, num_knot_v, xi, eta, knot_vector_u, knot_vector_v, weight):
        N = ti.Vector.zero(float, (self.basis_u.degree + 1) * (self.basis_v.degree + 1))
        num_ctrlpts_u = num_knot_u - 1 - self.basis_u.degree
        num_ctrlpts_v = num_knot_v - 1 - self.basis_v.degree

        spanU = self.basis_u.FindSpan(start_knot_u, num_ctrlpts_u, xi, knot_vector_u)
        spanV = self.basis_v.FindSpan(start_knot_v, num_ctrlpts_v, eta, knot_vector_v)
        NurbsU = self.basis_u.BasisFuncs(start_knot_u, spanU, xi, knot_vector_u)
        NurbsV = self.basis_v.BasisFuncs(start_knot_v, spanV, eta, knot_vector_v)

        w, kk = 0., 0
        uind = spanU - self.basis_u.degree
        for j in ti.static(range(self.basis_v.degree + 1)):
            vind = spanV - self.basis_v.degree + j
            for k in ti.static(range(self.basis_u.degree + 1)):
                linear_id = uind + k + vind * num_ctrlpts_u + start_ctrlpt
                nmp = NurbsU[k] * NurbsV[j]
                w += nmp * weight[linear_id]
                N[kk] = nmp * weight[linear_id]
                kk += 1
        return N / w

    @ti.func
    def NurbsBasisDers2d(self, start_knot_u, start_knot_v, start_ctrlpt, num_knot_u, num_knot_v, xi, eta, knot_vector_u, knot_vector_v, weight):
        N = ti.Vector.zero(float, (self.basis_u.degree + 1) * (self.basis_v.degree + 1))
        dNdnat = ti.Matrix.zero(float, (self.basis_u.degree + 1) * (self.basis_v.degree + 1), 2)
        num_ctrlpts_u = num_knot_u - 1 - self.basis_u.degree 
        num_ctrlpts_v = num_knot_v - 1 - self.basis_v.degree

        spanU = self.basis_u.FindSpan(start_knot_u, num_ctrlpts_u, xi, knot_vector_u)
        spanV = self.basis_v.FindSpan(start_knot_v, num_ctrlpts_v, eta, knot_vector_v)
        dNurbsU = self.basis_u.FirstDersBasisFuncs(start_knot_u, spanU, xi, knot_vector_u)
        dNurbsV = self.basis_v.FirstDersBasisFuncs(start_knot_v, spanV, eta, knot_vector_v)

        w, dwdxi, dwdet = 0., 0., 0.
        uind = spanU - self.basis_u.degree
        for j in ti.static(range(self.basis_v.degree + 1)):
            vind = spanV - self.basis_v.degree + j
            for k in ti.static(range(self.basis_u.degree + 1)):
                linear_id = uind + k + vind * num_ctrlpts_u + start_ctrlpt
                wgt = weight[linear_id]
                w += dNurbsU[0, k] * dNurbsV[0, j] * wgt
                dwdxi += dNurbsU[1, k] * dNurbsV[0, j] * wgt
                dwdet += dNurbsU[0, k] * dNurbsV[1, j] * wgt

        kk = 0
        uind = spanU - self.basis_u.degree
        for j in ti.static(range(self.basis_v.degree + 1)):
            vind = spanV - self.basis_v.degree + j
            for k in ti.static(range(self.basis_u.degree + 1)):
                linear_id = uind + k + vind * num_ctrlpts_u + start_ctrlpt
                fac = weight[linear_id] / (w * w)
                nmp = dNurbsU[0, k] * dNurbsV[0, j]
                N[kk] = nmp * fac * w
                dNdnat[kk, 0] = (dNurbsU[1, k] * dNurbsV[0, j] * w - nmp * dwdxi) * fac
                dNdnat[kk, 1] = (dNurbsU[0, k] * dNurbsV[1, j] * w - nmp * dwdet) * fac
                kk += 1
        return N, dNdnat

    @ti.func
    def NurbsBasis2ndDers2d(self, start_knot_u, start_knot_v, start_ctrlpt, num_knot_u, num_knot_v, xi, eta, knot_vector_u, knot_vector_v, weight):
        N = ti.Vector.zero(float, (self.basis_u.degree + 1) * (self.basis_v.degree + 1))
        dNdnat = ti.Matrix.zero(float, (self.basis_u.degree + 1) * (self.basis_v.degree + 1), 2)
        d2Ndnat = ti.Matrix.zero(float, (self.basis_u.degree + 1) * (self.basis_v.degree + 1), 3)
        num_ctrlpts_u = num_knot_u - 1 - self.basis_u.degree 
        num_ctrlpts_v = num_knot_v - 1 - self.basis_v.degree

        spanU = self.basis_u.FindSpan(start_knot_u, num_ctrlpts_u, xi, knot_vector_u)
        spanV = self.basis_v.FindSpan(start_knot_v, num_ctrlpts_v, eta, knot_vector_v)
        NurbsU = self.basis_u.SecondDersBasisFuncs(start_knot_u, spanU, xi, knot_vector_u)
        NurbsV = self.basis_v.SecondDersBasisFuncs(start_knot_v, spanV, eta, knot_vector_v)

        w = 0.0
        dwdxi, dwdet = 0.0, 0.0
        d2wdxi, d2wdet, d2wdxe = 0.0, 0.0, 0.0
        uind = spanU - self.basis_u.degree
        for j in ti.static(range(self.basis_v.degree + 1)):
            vind = spanV - self.basis_v.degree + j
            for k in ti.static(range(self.basis_u.degree + 1)):
                linear_id = uind + k + vind * num_ctrlpts_u + start_ctrlpt
                wgt = weight[linear_id]
                w += NurbsU[0, k] * NurbsV[0, j] * wgt
                dwdxi += NurbsU[1, k] * NurbsV[0, j] * wgt
                dwdet += NurbsU[0, k] * NurbsV[1, j] * wgt
                d2wdxi += NurbsU[2, k] * NurbsV[0, j] * wgt
                d2wdet += NurbsU[0, k] * NurbsV[2, j] * wgt
                d2wdxe += NurbsU[1, k] * NurbsV[1, j] * wgt

        kk = 0
        inv_w = 1 / w
        inv_w2 = 1 / w / w
        inv_w3 = 1 / w / w / w
        uind = spanU - self.basis_u.degree
        for j in ti.static(range(self.basis_v.degree + 1)):
            vind = spanV - self.basis_v.degree + j
            for k in ti.static(range(self.basis_u.degree + 1)):
                linear_id = uind + k + vind * num_ctrlpts_u + start_ctrlpt
                wgt = weight[linear_id]
                fac = wgt / (w * w)
                nmp = NurbsU[0, k] * NurbsV[0, j]

                N[kk] = nmp * fac * w
                dNdnat[kk, 0] = (NurbsU[1, k] * NurbsV[0, j] * w - nmp * dwdxi) * fac
                dNdnat[kk, 1] = (NurbsU[0, k] * NurbsV[1, j] * w - nmp * dwdet) * fac
                d2Ndnat[kk, 0] = (NurbsU[2, k] * NurbsV[0, j] * inv_w - 2 * NurbsU[1, k] * NurbsV[0, j] * dwdxi * inv_w2 - NurbsU[0, k] * NurbsV[0, j] * d2wdxi * inv_w2 + 2 * NurbsU[0, k] * NurbsV[0, j] * dwdxi * dwdxi * inv_w3) * wgt
                d2Ndnat[kk, 1] = (NurbsU[0, k] * NurbsV[2, j] * inv_w - 2 * NurbsU[0, k] * NurbsV[1, j] * dwdet * inv_w2 - NurbsU[0, k] * NurbsV[0, j] * d2wdet * inv_w2 + 2 * NurbsU[0, k] * NurbsV[0, j] * dwdet * dwdet * inv_w3) * wgt
                d2Ndnat[kk, 2] = (NurbsU[1, k]*NurbsV[1, j] * inv_w - NurbsU[1, k] * NurbsV[0, j] * dwdet * inv_w2 - NurbsU[0, k] * NurbsV[1, j] * dwdxi * inv_w2 - NurbsU[0, k] * NurbsV[0, j] * d2wdxe * inv_w2 + 2 * NurbsU[0, k] * NurbsV[0, j] * dwdxi * dwdet * inv_w3) * wgt
        return N, dNdnat, d2Ndnat

    @ti.func
    def NurbsBasisLocalInterpolations2d(self, start_knot_u, start_knot_v, spanU, spanV, xi, eta, knot_vector_u, knot_vector_v, point, weight):
        NurbsU = self.basis_u.BasisFuncs(start_knot_u, spanU, xi, knot_vector_u)
        NurbsV = self.basis_v.BasisFuncs(start_knot_v, spanV, eta, knot_vector_v)

        position, wght = ti.Vector.zero(float, self.dimension), 0.
        for j in ti.static(range(self.basis_v.degree + 1)):
            for k in ti.static(range(self.basis_u.degree + 1)):
                linear_id = k + j * (self.basis_u.degree + 1)
                nmp = NurbsU[k] * NurbsV[j]
                position += nmp * ti.Vector([point[linear_id, d] for d in ti.static(range(self.dimension))]) * weight[linear_id]
                wght += nmp * weight[linear_id]
        return position / wght

    @ti.func
    def NurbsBasisInterpolations2d(self, start_knot_u, start_knot_v, start_ctrlpt, num_knot_u, num_knot_v, xi, eta, knot_vector_u, knot_vector_v, point, weight):
        num_ctrlpts_u = num_knot_u - 1 - self.basis_u.degree 
        num_ctrlpts_v = num_knot_v - 1 - self.basis_v.degree

        spanU = self.basis_u.FindSpan(start_knot_u, num_ctrlpts_u, xi, knot_vector_u)
        spanV = self.basis_v.FindSpan(start_knot_v, num_ctrlpts_v, eta, knot_vector_v)
        NurbsU = self.basis_u.BasisFuncs(start_knot_u, spanU, xi, knot_vector_u)
        NurbsV = self.basis_v.BasisFuncs(start_knot_v, spanV, eta, knot_vector_v)

        position, wght = ti.Vector.zero(float, self.dimension), 0.
        uind = spanU - self.basis_u.degree
        for j in ti.static(range(self.basis_v.degree + 1)):
            vind = spanV - self.basis_v.degree + j
            for k in ti.static(range(self.basis_u.degree + 1)):
                linear_id = uind + k + vind * num_ctrlpts_u + start_ctrlpt
                nmp = NurbsU[k] * NurbsV[j]
                position += nmp * point[linear_id] * weight[linear_id]
                wght += nmp * weight[linear_id]
        return position / wght

    @ti.func
    def NurbsBasisInterpolationsDers2d(self, start_knot_u, start_knot_v, start_ctrlpt, num_knot_u, num_knot_v, xi, eta, knot_vector_u, knot_vector_v, point, weight):
        num_ctrlpts_u = num_knot_u - 1 - self.basis_u.degree 
        num_ctrlpts_v = num_knot_v - 1 - self.basis_v.degree

        spanU = self.basis_u.FindSpan(start_knot_u, num_ctrlpts_u, xi, knot_vector_u)
        spanV = self.basis_v.FindSpan(start_knot_v, num_ctrlpts_v, eta, knot_vector_v)
        dNurbsU = self.basis_u.FirstDersBasisFuncs(start_knot_u, spanU, xi, knot_vector_u)
        dNurbsV = self.basis_v.FirstDersBasisFuncs(start_knot_v, spanV, eta, knot_vector_v)

        w, dwdxi, dwdet = 0., 0., 0.
        uind = spanU - self.basis_u.degree
        for j in ti.static(range(self.basis_v.degree + 1)):
            vind = spanV - self.basis_v.degree + j
            for k in ti.static(range(self.basis_u.degree + 1)):
                linear_id = uind + k + vind * num_ctrlpts_u + start_ctrlpt
                wgt = weight[linear_id]
                w += dNurbsU[0, k] * dNurbsV[0, j] * wgt
                dwdxi += dNurbsU[1, k] * dNurbsV[0, j] * wgt
                dwdet += dNurbsU[0, k] * dNurbsV[1, j] * wgt

        position, dirsU, dirsV = ti.Vector.zero(float, self.dimension), ti.Vector.zero(float, self.dimension), ti.Vector.zero(float, self.dimension)
        uind = spanU - self.basis_u.degree
        for j in ti.static(range(self.basis_v.degree + 1)):
            vind = spanV - self.basis_v.degree + j
            for k in ti.static(range(self.basis_u.degree + 1)):
                linear_id = uind + k + vind * num_ctrlpts_u + start_ctrlpt
                fac = weight[linear_id] / (w * w)
                nmp = dNurbsU[0, k] * dNurbsV[0, j]
                position += nmp * fac * point[linear_id] * w
                dirsU += (dNurbsU[1, k] * dNurbsV[0, k] * w - nmp * dwdxi) * fac * point[linear_id]
                dirsV += (dNurbsU[0, k] * dNurbsV[1, k] * w - nmp * dwdet) * fac * point[linear_id]
        return position, dirsU, dirsV

    @ti.func
    def NurbsBasisInterpolations2ndDers2d(self, start_knot_u, start_knot_v, start_ctrlpt, num_knot_u, num_knot_v, xi, eta, knot_vector_u, knot_vector_v, point, weight):
        num_ctrlpts_u = num_knot_u - 1 - self.basis_u.degree 
        num_ctrlpts_v = num_knot_v - 1 - self.basis_v.degree

        spanU = self.basis_u.FindSpan(start_knot_u, num_ctrlpts_u, xi, knot_vector_u)
        spanV = self.basis_v.FindSpan(start_knot_v, num_ctrlpts_v, eta, knot_vector_v)
        NurbsU = self.basis_u.SecondDersBasisFuncs(start_knot_u, spanU, xi, knot_vector_u)
        NurbsV = self.basis_v.SecondDersBasisFuncs(start_knot_v, spanV, eta, knot_vector_v)

        w = 0.0
        dwdxi, dwdet = 0.0, 0.0
        d2wdxi, d2wdet, d2wdxe = 0.0, 0.0, 0.0
        uind = spanU - self.basis_u.degree
        for j in ti.static(range(self.basis_v.degree + 1)):
            vind = spanV - self.basis_v.degree + j
            for k in ti.static(range(self.basis_u.degree + 1)):
                linear_id = uind + k + vind * num_ctrlpts_u + start_ctrlpt
                wgt = weight[linear_id]
                w += NurbsU[0, k] * NurbsV[0, j] * wgt
                dwdxi += NurbsU[1, k] * NurbsV[0, j] * wgt
                dwdet += NurbsU[0, k] * NurbsV[1, j] * wgt
                d2wdxi += NurbsU[2, k] * NurbsV[0, j] * wgt
                d2wdet += NurbsU[0, k] * NurbsV[2, j] * wgt
                d2wdxe += NurbsU[1, k] * NurbsV[1, j] * wgt

        position, dirsU, dirsV = ti.Vector.zero(float, self.dimension), ti.Vector.zero(float, self.dimension), ti.Vector.zero(float, self.dimension)
        ddirsUU, ddirsVV, ddirsUV = ti.Vector.zero(float, self.dimension), ti.Vector.zero(float, self.dimension), ti.Vector.zero(float, self.dimension)
        # dirsU -> dNdxi, dirsV -> dNdeta
        # ddirsUU -> d2Ndxi2, ddirsVV -> d2Ndeta2, ddirsUV -> d2Ndxideta
        inv_w = 1 / w
        inv_w2 = 1 / w / w
        inv_w3 = 1 / w / w / w
        uind = spanU - self.basis_u.degree
        for j in ti.static(range(self.basis_v.degree + 1)):
            vind = spanV - self.basis_v.degree + j
            for k in ti.static(range(self.basis_u.degree + 1)):
                linear_id = uind + k + vind * num_ctrlpts_u + start_ctrlpt
                wgt = weight[linear_id]
                fac = wgt / (w * w)
                nmp = NurbsU[0, k] * NurbsV[0, j]

                position += nmp * fac * w * point[linear_id]
                dirsU += (NurbsU[1, k] * NurbsV[0, j] * w - nmp * dwdxi) * fac * point[linear_id]
                dirsV += (NurbsU[0, k] * NurbsV[1, j] * w - nmp * dwdet) * fac * point[linear_id]
                ddirsUU += (NurbsU[2, k] * NurbsV[0, j] * inv_w - 2 * NurbsU[1, k] * NurbsV[0, j] * dwdxi * inv_w2 - NurbsU[0, k] * NurbsV[0, j] * d2wdxi * inv_w2 + 2 * NurbsU[0, k] * NurbsV[0, j] * dwdxi * dwdxi * inv_w3) * wgt * point[linear_id]
                ddirsVV += (NurbsU[0, k] * NurbsV[2, j] * inv_w - 2 * NurbsU[0, k] * NurbsV[1, j] * dwdet * inv_w2 - NurbsU[0, k] * NurbsV[0, j] * d2wdet * inv_w2 + 2 * NurbsU[0, k] * NurbsV[0, j] * dwdet * dwdet * inv_w3) * wgt * point[linear_id]
                ddirsUV += (NurbsU[1, k]*NurbsV[1, j] * inv_w - NurbsU[1, k] * NurbsV[0, j] * dwdet * inv_w2 - NurbsU[0, k] * NurbsV[1, j] * dwdxi * inv_w2 - NurbsU[0, k] * NurbsV[0, j] * d2wdxe * inv_w2 + 2 * NurbsU[0, k] * NurbsV[0, j] * dwdxi * dwdet * inv_w3) * wgt * point[linear_id]
        return position, dirsU, dirsV, ddirsUU, ddirsVV, ddirsUV
    
    @ti.func
    def NurbsSpan(self, start_knot_u, start_knot_v, num_knot_u, num_knot_v, xi, eta, knot_vector_u, knot_vector_v):
        num_ctrlpts_u = num_knot_u - 1 - self.basis_u.degree 
        num_ctrlpts_v = num_knot_v - 1 - self.basis_v.degree

        spanU = self.basis_u.FindSpan(start_knot_u, num_ctrlpts_u, xi, knot_vector_u)
        spanV = self.basis_v.FindSpan(start_knot_v, num_ctrlpts_v, eta, knot_vector_v)
        return spanU, spanV
    
    @ti.func
    def NurbsBasisShape(self, start_knot_u, start_knot_v, start_ctrlpt, num_knot_u, num_knot_v, xi, eta, knot_vector_u, knot_vector_v, weight):
        num_ctrlpts_u = num_knot_u - 1 - self.basis_u.degree 
        num_ctrlpts_v = num_knot_v - 1 - self.basis_v.degree

        spanU = self.basis_u.FindSpan(start_knot_u, num_ctrlpts_u, xi, knot_vector_u)
        spanV = self.basis_v.FindSpan(start_knot_v, num_ctrlpts_v, eta, knot_vector_v)
        NurbsU = self.basis_u.SecondDersBasisFuncs(start_knot_u, spanU, xi, knot_vector_u)
        NurbsV = self.basis_v.SecondDersBasisFuncs(start_knot_v, spanV, eta, knot_vector_v)

        w = 0.0
        uind = spanU - self.basis_u.degree
        for j in ti.static(range(self.basis_v.degree + 1)):
            vind = spanV - self.basis_v.degree + j
            for k in ti.static(range(self.basis_u.degree + 1)):
                linear_id = uind + k + vind * num_ctrlpts_u + start_ctrlpt
                wgt = weight[linear_id]
                w += NurbsU[0, k] * NurbsV[0, j] * wgt

        N = ti.Vector.zero(float, (self.basis_u.degree + 1) * (self.basis_v.degree + 1))
        kk = 0
        uind = spanU - self.basis_u.degree
        for j in ti.static(range(self.basis_v.degree + 1)):
            vind = spanV - self.basis_v.degree + j
            for k in ti.static(range(self.basis_u.degree + 1)):
                linear_id = uind + k + vind * num_ctrlpts_u + start_ctrlpt
                wgt = weight[linear_id]
                fac = wgt / (w * w)
                nmp = NurbsU[0, k] * NurbsV[0, j]

                nshape = nmp * fac * w
                N[kk] = nshape
                kk += 1
        return spanU, spanV, N
    
    @ti.func
    def NurbsBasisHessian(self, start_knot_u, start_knot_v, start_ctrlpt, num_knot_u, num_knot_v, xi, eta, knot_vector_u, knot_vector_v, point, weight):
        num_ctrlpts_u = num_knot_u - 1 - self.basis_u.degree 
        num_ctrlpts_v = num_knot_v - 1 - self.basis_v.degree

        spanU = self.basis_u.FindSpan(start_knot_u, num_ctrlpts_u, xi, knot_vector_u)
        spanV = self.basis_v.FindSpan(start_knot_v, num_ctrlpts_v, eta, knot_vector_v)
        NurbsU = self.basis_u.SecondDersBasisFuncs(start_knot_u, spanU, xi, knot_vector_u)
        NurbsV = self.basis_v.SecondDersBasisFuncs(start_knot_v, spanV, eta, knot_vector_v)

        w = 0.0
        dwdxi, dwdet = 0.0, 0.0
        d2wdxi, d2wdet, d2wdxe = 0.0, 0.0, 0.0
        uind = spanU - self.basis_u.degree
        for j in ti.static(range(self.basis_v.degree + 1)):
            vind = spanV - self.basis_v.degree + j
            for k in ti.static(range(self.basis_u.degree + 1)):
                linear_id = uind + k + vind * num_ctrlpts_u + start_ctrlpt
                wgt = weight[linear_id]
                w += NurbsU[0, k] * NurbsV[0, j] * wgt
                dwdxi += NurbsU[1, k] * NurbsV[0, j] * wgt
                dwdet += NurbsU[0, k] * NurbsV[1, j] * wgt
                d2wdxi += NurbsU[2, k] * NurbsV[0, j] * wgt
                d2wdet += NurbsU[0, k] * NurbsV[2, j] * wgt
                d2wdxe += NurbsU[1, k] * NurbsV[1, j] * wgt

        position, dirsU, dirsV = ti.Vector.zero(float, self.dimension), ti.Vector.zero(float, self.dimension), ti.Vector.zero(float, self.dimension)
        ddirsUU, ddirsVV, ddirsUV = ti.Vector.zero(float, self.dimension), ti.Vector.zero(float, self.dimension), ti.Vector.zero(float, self.dimension)
        N = ti.Vector.zero(float, (self.basis_u.degree + 1) * (self.basis_v.degree + 1))
        dNdnatU = ti.Vector.zero(float, (self.basis_u.degree + 1) * (self.basis_v.degree + 1))
        dNdnatV = ti.Vector.zero(float, (self.basis_u.degree + 1) * (self.basis_v.degree + 1))
        d2NdnatUU = ti.Vector.zero(float, (self.basis_u.degree + 1) * (self.basis_v.degree + 1))
        d2NdnatVV = ti.Vector.zero(float, (self.basis_u.degree + 1) * (self.basis_v.degree + 1))
        d2NdnatUV = ti.Vector.zero(float, (self.basis_u.degree + 1) * (self.basis_v.degree + 1))
        kk = 0
        # dirsU -> dNdxi, dirsV -> dNdeta
        # ddirsUU -> d2Ndxi2, ddirsVV -> d2Ndeta2, ddirsUV -> d2Ndxideta
        inv_w = 1 / w
        inv_w2 = 1 / w / w
        inv_w3 = 1 / w / w / w
        uind = spanU - self.basis_u.degree
        for j in ti.static(range(self.basis_v.degree + 1)):
            vind = spanV - self.basis_v.degree + j
            for k in ti.static(range(self.basis_u.degree + 1)):
                linear_id = uind + k + vind * num_ctrlpts_u + start_ctrlpt
                wgt = weight[linear_id]
                fac = wgt / (w * w)
                nmp = NurbsU[0, k] * NurbsV[0, j]

                nshape = nmp * fac * w
                derivate_u = (NurbsU[1, k] * NurbsV[0, j] * w - nmp * dwdxi) * fac
                derivate_v = (NurbsU[0, k] * NurbsV[1, j] * w - nmp * dwdet) * fac
                dderivate_uu = (NurbsU[2, k] * NurbsV[0, j] * inv_w - 2 * NurbsU[1, k] * NurbsV[0, j] * dwdxi * inv_w2 - NurbsU[0, k] * NurbsV[0, j] * d2wdxi * inv_w2 + 2 * NurbsU[0, k] * NurbsV[0, j] * dwdxi * dwdxi * inv_w3) * wgt
                dderivate_vv = (NurbsU[0, k] * NurbsV[2, j] * inv_w - 2 * NurbsU[0, k] * NurbsV[1, j] * dwdet * inv_w2 - NurbsU[0, k] * NurbsV[0, j] * d2wdet * inv_w2 + 2 * NurbsU[0, k] * NurbsV[0, j] * dwdet * dwdet * inv_w3) * wgt
                dderivate_uv = (NurbsU[1, k]*NurbsV[1, j] * inv_w - NurbsU[1, k] * NurbsV[0, j] * dwdet * inv_w2 - NurbsU[0, k] * NurbsV[1, j] * dwdxi * inv_w2 - NurbsU[0, k] * NurbsV[0, j] * d2wdxe * inv_w2 + 2 * NurbsU[0, k] * NurbsV[0, j] * dwdxi * dwdet * inv_w3) * wgt

                position += nshape * point[linear_id]
                dirsU += derivate_u * point[linear_id]
                dirsV += derivate_v * point[linear_id]
                ddirsUU += dderivate_uu * point[linear_id]
                ddirsVV += dderivate_vv * point[linear_id]
                ddirsUV += dderivate_uv * point[linear_id]
        
                N[kk] = nshape
                dNdnatU[kk] = derivate_u * fac
                dNdnatV[kk] = derivate_v * fac
                d2NdnatUU[kk] = dderivate_uu
                d2NdnatVV[kk] = dderivate_vv
                d2NdnatUV[kk] = dderivate_uv
                kk += 1
        return spanU, spanV, N, dNdnatU, dNdnatV, dirsU, dirsV, ddirsUU, ddirsVV, ddirsUV


@ti.data_oriented
class NurbsBasisFunction3d:
    def __init__(self, degree_u, degree_v, degree_w):
        self.dimension = 3
        self.basis_u = NurbsBasisFunction(degree_u)
        self.basis_v = NurbsBasisFunction(degree_v)
        self.basis_w = NurbsBasisFunction(degree_w)

    @ti.func
    def FindElement(self, start_knot_u, start_knot_v, start_knot_w, num_knot_u, num_knot_v, num_knot_w, xi, eta, zeta, knot_vector_u, knot_vector_v, knot_vector_w):
        num_ctrlpts_u = num_knot_u - 1 - self.basis_u.degree 
        num_ctrlpts_v = num_knot_v - 1 - self.basis_v.degree 
        num_ctrlpts_w = num_knot_w - 1 - self.basis_w.degree 

        spanU = self.basis_u.FindSpan(start_knot_u, num_ctrlpts_u, xi, knot_vector_u)
        spanV = self.basis_v.FindSpan(start_knot_v, num_ctrlpts_v, eta, knot_vector_v)
        spanW = self.basis_w.FindSpan(start_knot_w, num_ctrlpts_w, zeta, knot_vector_w)
        return spanU - self.basis_u.degree, spanV - self.basis_v.degree, spanW - self.basis_w.degree

    @ti.func
    def NurbsBasis3d(self, start_knot_u, start_knot_v, start_knot_w, start_ctrlpt, num_knot_u, num_knot_v, num_knot_w, xi, eta, zeta, knot_vector_u, knot_vector_v, knot_vector_w, weight):
        N = ti.Vector.zero(float, (self.basis_u.degree + 1) * (self.basis_v.degree + 1) * (self.basis_w.degree + 1))
        num_ctrlpts_u = num_knot_u - 1 - self.basis_u.degree 
        num_ctrlpts_v = num_knot_v - 1 - self.basis_v.degree 
        num_ctrlpts_w = num_knot_w - 1 - self.basis_w.degree 

        spanU = self.basis_u.FindSpan(start_knot_u, num_ctrlpts_u, xi, knot_vector_u)
        spanV = self.basis_v.FindSpan(start_knot_v, num_ctrlpts_v, eta, knot_vector_v)
        spanW = self.basis_w.FindSpan(start_knot_w, num_ctrlpts_w, zeta, knot_vector_w)
        NurbsU = self.basis_u.BasisFuncs(start_knot_u, spanU, xi, knot_vector_u)
        NurbsV = self.basis_v.BasisFuncs(start_knot_v, spanV, eta, knot_vector_v)
        NurbsW = self.basis_w.BasisFuncs(start_knot_w, spanW, zeta, knot_vector_w)

        w, kk = 0., 0
        uind = spanU - self.basis_u.degree
        for i in ti.static(range(self.basis_w.degree + 1)):
            wind = spanW - self.basis_w.degree + i
            for j in ti.static(range(self.basis_v.degree + 1)):
                vind = spanV - self.basis_v.degree + j
                for k in ti.static(range(self.basis_u.degree + 1)):
                    linear_id = uind + k + vind * num_ctrlpts_u + wind * num_ctrlpts_u * num_ctrlpts_v + start_ctrlpt
                    nmp = NurbsU[k] * NurbsV[j] * NurbsW[i]
                    w += nmp * weight[linear_id]
                    N[kk] = nmp * weight[linear_id]
                    kk += 1
        return N / w

    @ti.func
    def NurbsBasisDers3d(self, start_knot_u, start_knot_v, start_knot_w, start_ctrlpt, num_knot_u, num_knot_v, num_knot_w, xi, eta, zeta, knot_vector_u, knot_vector_v, knot_vector_w, weight):
        N = ti.Vector.zero(float, (self.basis_u.degree + 1) * (self.basis_v.degree + 1) * (self.basis_w.degree + 1))
        dNdnat = ti.Matrix.zero(float, (self.basis_u.degree + 1) * (self.basis_v.degree + 1) * (self.basis_w.degree + 1), 3)
        num_ctrlpts_u = num_knot_u - 1 - self.basis_u.degree 
        num_ctrlpts_v = num_knot_v - 1 - self.basis_v.degree 
        num_ctrlpts_w = num_knot_w - 1 - self.basis_w.degree 

        spanU = self.basis_u.FindSpan(start_knot_u, num_ctrlpts_u, xi, knot_vector_u)
        spanV = self.basis_v.FindSpan(start_knot_v, num_ctrlpts_v, eta, knot_vector_v)
        spanW = self.basis_w.FindSpan(start_knot_w, num_ctrlpts_w, zeta, knot_vector_w)
        dNurbsU = self.basis_u.FirstDersBasisFuncs(start_knot_u, spanU, xi, knot_vector_u)
        dNurbsV = self.basis_v.FirstDersBasisFuncs(start_knot_v, spanV, eta, knot_vector_v)
        dNurbsW = self.basis_w.FirstDersBasisFuncs(start_knot_w, spanW, zeta, knot_vector_w)

        w, dwdxi, dwdet, dwdze = 0., 0., 0., 0.
        uind = spanU - self.basis_u.degree
        for i in ti.static(range(self.basis_w.degree + 1)):
            wind = spanW - self.basis_w.degree + i
            for j in ti.static(range(self.basis_v.degree + 1)):
                vind = spanV - self.basis_v.degree + j
                for k in ti.static(range(self.basis_u.degree + 1)):
                    linear_id = uind + k + vind * num_ctrlpts_u + wind * num_ctrlpts_u * num_ctrlpts_v + start_ctrlpt
                    wgt = weight[linear_id]
                    w += dNurbsU[0, k] * dNurbsV[0, j] * dNurbsW[0, i] * wgt
                    dwdxi += dNurbsU[1, k] * dNurbsV[0, j] * dNurbsW[0, i] * wgt
                    dwdet += dNurbsU[0, k] * dNurbsV[1, j] * dNurbsW[0, i] * wgt
                    dwdze += dNurbsU[0, k] * dNurbsV[0, j] * dNurbsW[1, i] * wgt

        kk = 0
        uind = spanU - self.basis_u.degree
        for i in ti.static(range(self.basis_w.degree + 1)):
            wind = spanW - self.basis_w.degree + i
            for j in ti.static(range(self.basis_v.degree + 1)):
                vind = spanV - self.basis_v.degree + j
                for k in ti.static(range(self.basis_u.degree + 1)):
                    linear_id = uind + k + vind * num_ctrlpts_u + wind * num_ctrlpts_u * num_ctrlpts_v + start_ctrlpt
                    fac = weight[linear_id] / (w * w)
                    nmp = dNurbsU[0, k] * dNurbsV[0, j] * dNurbsW[0, i]
                    N[kk] = nmp * fac * w
                    dNdnat[kk, 0] = (dNurbsU[1, k] * dNurbsV[0, j] * dNurbsW[0, i] * w - nmp * dwdxi) * fac
                    dNdnat[kk, 1] = (dNurbsU[0, k] * dNurbsV[1, j] * dNurbsW[0, i] * w - nmp * dwdet) * fac
                    dNdnat[kk, 2] = (dNurbsU[0, k] * dNurbsV[0, j] * dNurbsW[1, i] * w - nmp * dwdze) * fac
                    kk += 1
        return N, dNdnat

    @ti.func
    def NurbsBasis2ndDers3d(self, start_knot_u, start_knot_v, start_knot_w, start_ctrlpt, num_knot_u, num_knot_v, num_knot_w, xi, eta, zeta, knot_vector_u, knot_vector_v, knot_vector_w, weight):
        N = ti.Vector.zero(float, (self.basis_u.degree + 1) * (self.basis_v.degree + 1) * (self.basis_w.degree + 1))
        dNdnat = ti.Matrix.zero(float, (self.basis_u.degree + 1) * (self.basis_v.degree + 1) * (self.basis_w.degree + 1), 3)
        d2Ndnat = ti.Matrix.zero(float, (self.basis_u.degree + 1) * (self.basis_v.degree + 1) * (self.basis_w.degree + 1), 6)
        num_ctrlpts_u = num_knot_u - 1 - self.basis_u.degree 
        num_ctrlpts_v = num_knot_v - 1 - self.basis_v.degree 
        num_ctrlpts_w = num_knot_w - 1 - self.basis_w.degree 

        spanU = self.basis_u.FindSpan(start_knot_u, num_ctrlpts_u, xi, knot_vector_u)
        spanV = self.basis_v.FindSpan(start_knot_v, num_ctrlpts_v, eta, knot_vector_v)
        spanW = self.basis_w.FindSpan(start_knot_w, num_ctrlpts_w, zeta, knot_vector_w)
        NurbsU = self.basis_u.SecondDersBasisFuncs(start_knot_u, spanU, xi, knot_vector_u)
        NurbsV = self.basis_v.SecondDersBasisFuncs(start_knot_v, spanV, eta, knot_vector_v)
        NurbsW = self.basis_w.SecondDersBasisFuncs(start_knot_w, spanW, zeta, knot_vector_w)

        w     = 0.0
        dwdxi, dwdet, dwdze = 0.0, 0.0, 0.0
        d2wdxi, d2wdet, d2wdze = 0.0, 0.0, 0.0
        d2wdxe, d2wdez, d2wdxz = 0.0, 0.0, 0.0
        uind = spanU - self.basis_u.degree
        for i in ti.static(range(self.basis_w.degree + 1)):
            wind = spanW - self.basis_w.degree + i
            for j in ti.static(range(self.basis_v.degree + 1)):
                vind = spanV - self.basis_v.degree + j
                for k in ti.static(range(self.basis_u.degree + 1)):
                    linear_id = uind + k + vind * num_ctrlpts_u + wind * num_ctrlpts_u * num_ctrlpts_v + start_ctrlpt
                    wgt = weight[linear_id]

                    w += NurbsU[0, k] * NurbsV[0, j] * NurbsW[0, i] * wgt
                    dwdxi += NurbsU[1, k] * NurbsV[0, j] * NurbsW[0, i] * wgt
                    dwdet += NurbsU[0, k] * NurbsV[1, j] * NurbsW[0, i] * wgt
                    dwdze += NurbsU[0, k] * NurbsV[0, j] * NurbsW[1, i] * wgt
                    d2wdxi += NurbsU[2, k] * NurbsV[0, j] * NurbsW[0, i] * wgt
                    d2wdet += NurbsU[0, k] * NurbsV[2, j] * NurbsW[0, i] * wgt
                    d2wdze += NurbsU[0, k] * NurbsV[0, j] * NurbsW[2, i] * wgt
                    d2wdxe += NurbsU[1, k] * NurbsV[1, j] * wgt
                    d2wdez += NurbsV[1, j] * NurbsW[1, i] * wgt
                    d2wdxz += NurbsU[1, k] * NurbsW[1, i] * wgt

        kk = 0
        inv_w = 1 / w
        inv_w2 = 1 / w / w
        inv_w3 = 1 / w / w / w
        uind = spanU - self.basis_u.degree
        for i in ti.static(range(self.basis_w.degree + 1)):
            wind = spanW - self.basis_w.degree + i
            for j in ti.static(range(self.basis_v.degree + 1)):
                vind = spanV - self.basis_v.degree + j
                for k in ti.static(range(self.basis_u.degree + 1)):
                    linear_id = uind + k + vind * num_ctrlpts_u + wind * num_ctrlpts_u * num_ctrlpts_v + start_ctrlpt
                    wgt = weight[linear_id]
                    fac = wgt / (w * w)
                    nmp = NurbsU[0, k] * NurbsV[0, j] * NurbsW[0, i]

                    N[kk] = nmp * fac * w
                    dNdnat[kk, 0] = (NurbsU[1, k] * NurbsV[0, j] * NurbsW[0, i] * w - nmp * dwdxi) * fac
                    dNdnat[kk, 1] = (NurbsU[0, k] * NurbsV[1, j] * NurbsW[0, i] * w - nmp * dwdet) * fac
                    dNdnat[kk, 2] = (NurbsU[0, k] * NurbsV[0, j] * NurbsW[1, i] * w - nmp * dwdze) * fac
                    d2Ndnat[kk, 0] = (NurbsU[2, k] * NurbsV[0, j] * NurbsW[0, i] * inv_w - 2 * NurbsU[1, k] * NurbsV[0, j] * NurbsW[0, i] * dwdxi * inv_w2 - NurbsU[0, k] * NurbsV[0, j] * NurbsW[0, i] * d2wdxi * inv_w2 + 2 * NurbsU[0, k] * NurbsV[0, j] * NurbsW[0, i] * dwdxi * dwdxi * inv_w3) * wgt
                    d2Ndnat[kk, 1] = (NurbsU[0, k] * NurbsV[2, j] * NurbsW[0, i] * inv_w - 2 * NurbsU[0, k] * NurbsV[1, j] * NurbsW[0, i] * dwdet * inv_w2 - NurbsU[0, k] * NurbsV[0, j] * NurbsW[0, i] * d2wdet * inv_w2 + 2 * NurbsU[0, k] * NurbsV[0, j] * NurbsW[0, i] * dwdet * dwdet * inv_w3) * wgt
                    d2Ndnat[kk, 2] = (NurbsU[0, k] * NurbsV[0, j] * NurbsW[2, i] * inv_w - 2 * NurbsU[0, k] * NurbsV[0, j] * NurbsW[1, i] * dwdze * inv_w2 - NurbsU[0, k] * NurbsV[0, j] * NurbsW[0, i] * d2wdze * inv_w2 + 2 * NurbsU[0, k] * NurbsV[0, j] * NurbsW[0, i] * dwdze * dwdze * inv_w3) * wgt
                    d2Ndnat[kk, 3] = (NurbsU[1, k]*NurbsV[1, j] * inv_w - NurbsU[1, k] * NurbsV[0, j] * dwdet * inv_w2 - NurbsU[0, k] * NurbsV[1, j] * dwdxi * inv_w2 - NurbsU[0, k] * NurbsV[0, j] * d2wdxe * inv_w2 + 2 * NurbsU[0, k] * NurbsV[0, j] * dwdxi * dwdet * inv_w3) * wgt
                    d2Ndnat[kk, 4] = (NurbsV[1, j]*NurbsW[1, i] * inv_w - NurbsV[1, j] * NurbsW[0, i] * dwdxi * inv_w2 - NurbsV[0, j] * NurbsW[1, i] * dwdet * inv_w2 - NurbsV[0, j] * NurbsW[0, i] * d2wdez * inv_w2 + 2 * NurbsV[0, j] * NurbsW[0, i] * dwdet * dwdze * inv_w3) * wgt
                    d2Ndnat[kk, 5] = (NurbsU[1, k]*NurbsW[1, i] * inv_w - NurbsW[1, i] * NurbsU[0, k] * dwdet * inv_w2 - NurbsW[0, i] * NurbsU[1, k] * dwdze * inv_w2 - NurbsU[0, k] * NurbsW[0, i] * d2wdxz * inv_w2 + 2 * NurbsU[0, k] * NurbsW[0, i] * dwdxi * dwdze * inv_w3) * wgt
        return N, dNdnat, d2Ndnat

    @ti.func
    def NurbsBasisInterpolations3d(self, start_knot_u, start_knot_v, start_knot_w, start_ctrlpt, num_knot_u, num_knot_v, num_knot_w, xi, eta, zeta, knot_vector_u, knot_vector_v, knot_vector_w, point, weight):
        num_ctrlpts_u = num_knot_u - 1 - self.basis_u.degree 
        num_ctrlpts_v = num_knot_v - 1 - self.basis_v.degree 
        num_ctrlpts_w = num_knot_w - 1 - self.basis_w.degree 

        spanU = self.basis_u.FindSpan(start_knot_u, num_ctrlpts_u, xi, knot_vector_u)
        spanV = self.basis_v.FindSpan(start_knot_v, num_ctrlpts_v, eta, knot_vector_v)
        spanW = self.basis_w.FindSpan(start_knot_w, num_ctrlpts_w, zeta, knot_vector_w)
        NurbsU = self.basis_u.BasisFuncs(start_knot_u, spanU, xi, knot_vector_u)
        NurbsV = self.basis_v.BasisFuncs(start_knot_v, spanV, eta, knot_vector_v)
        NurbsW = self.basis_w.BasisFuncs(start_knot_w, spanW, zeta, knot_vector_w)

        position, wght = ti.Vector.zero(float, self.dimension), 0.
        uind = spanU - self.basis_u.degree
        for i in ti.static(range(self.basis_w.degree + 1)):
            wind = spanW - self.basis_w.degree + i
            for j in ti.static(range(self.basis_v.degree + 1)):
                vind = spanV - self.basis_v.degree + j
                for k in ti.static(range(self.basis_u.degree + 1)):
                    linear_id = uind + k + vind * num_ctrlpts_u + wind * num_ctrlpts_u * num_ctrlpts_v + start_ctrlpt
                    nmp = NurbsU[k] * NurbsV[j] * NurbsW[i]
                    position += nmp * point[linear_id] * weight[linear_id]
                    wght += nmp * weight[linear_id]
        return position / wght

    @ti.func
    def NurbsBasisInterpolationsDers3d(self, start_knot_u, start_knot_v, start_knot_w, start_ctrlpt, num_knot_u, num_knot_v, num_knot_w, xi, eta, zeta, knot_vector_u, knot_vector_v, knot_vector_w, point, weight):
        num_ctrlpts_u = num_knot_u - 1 - self.basis_u.degree 
        num_ctrlpts_v = num_knot_v - 1 - self.basis_v.degree 
        num_ctrlpts_w = num_knot_w - 1 - self.basis_w.degree 

        spanU = self.basis_u.FindSpan(start_knot_u, num_ctrlpts_u, xi, knot_vector_u)
        spanV = self.basis_v.FindSpan(start_knot_v, num_ctrlpts_v, eta, knot_vector_v)
        spanW = self.basis_w.FindSpan(start_knot_w, num_ctrlpts_w, zeta, knot_vector_w)
        dNurbsU = self.basis_u.FirstDersBasisFuncs(start_knot_u, spanU, xi, knot_vector_u)
        dNurbsV = self.basis_v.FirstDersBasisFuncs(start_knot_v, spanV, eta, knot_vector_v)
        dNurbsW = self.basis_w.FirstDersBasisFuncs(start_knot_w, spanW, zeta, knot_vector_w)

        w, dwdxi, dwdet, dwdze = 0., 0., 0., 0.
        uind = spanU - self.basis_u.degree
        for i in ti.static(range(self.basis_w.degree + 1)):
            wind = spanW - self.basis_w.degree + i
            for j in ti.static(range(self.basis_v.degree + 1)):
                vind = spanV - self.basis_v.degree + j
                for k in ti.static(range(self.basis_u.degree + 1)):
                    linear_id = uind + k + vind * num_ctrlpts_u + wind * num_ctrlpts_u * num_ctrlpts_v + start_ctrlpt
                    wgt = weight[linear_id]
                    w += dNurbsU[0, k] * dNurbsV[0, j] * dNurbsW[0, i] * wgt
                    dwdxi += dNurbsU[1, k] * dNurbsV[0, j] * dNurbsW[0, i] * wgt
                    dwdet += dNurbsU[0, k] * dNurbsV[1, j] * dNurbsW[0, i] * wgt
                    dwdze += dNurbsU[0, k] * dNurbsV[0, j] * dNurbsW[1, i] * wgt

        position, dirsU, dirsV, dirsW = ti.Vector.zero(float, self.dimension), ti.Vector.zero(float, self.dimension), ti.Vector.zero(float, self.dimension), ti.Vector.zero(float, self.dimension)
        uind = spanU - self.basis_u.degree
        for i in ti.static(range(self.basis_w.degree + 1)):
            wind = spanW - self.basis_w.degree + i
            for j in ti.static(range(self.basis_v.degree + 1)):
                vind = spanV - self.basis_v.degree + j
                for k in ti.static(range(self.basis_u.degree + 1)):
                    linear_id = uind + k + vind * num_ctrlpts_u + wind * num_ctrlpts_u * num_ctrlpts_v + start_ctrlpt
                    fac = weight[linear_id] / (w * w)
                    nmp = dNurbsU[0, k] * dNurbsV[0, j] * dNurbsW[0, i]
                    position[0] += nmp * fac * w * point[linear_id]
                    dirsU += (dNurbsU[1, k] * dNurbsV[0, j] * dNurbsW[0, i] * w - nmp * dwdxi) * fac * point[linear_id]
                    dirsV += (dNurbsU[0, k] * dNurbsV[1, j] * dNurbsW[0, i] * w - nmp * dwdet) * fac * point[linear_id]
                    dirsW += (dNurbsU[0, k] * dNurbsV[0, j] * dNurbsW[1, i] * w - nmp * dwdze) * fac * point[linear_id]
        return position, dirsU, dirsV, dirsW

    @ti.func
    def NurbsBasisInterpolations2ndDers3d(self, start_knot_u, start_knot_v, start_knot_w, start_ctrlpt, num_knot_u, num_knot_v, num_knot_w, xi, eta, zeta, knot_vector_u, knot_vector_v, knot_vector_w, point, weight):
        num_ctrlpts_u = num_knot_u - 1 - self.basis_u.degree 
        num_ctrlpts_v = num_knot_v - 1 - self.basis_v.degree 
        num_ctrlpts_w = num_knot_w - 1 - self.basis_w.degree 

        spanU = self.basis_u.FindSpan(start_knot_u, num_ctrlpts_u, xi, knot_vector_u)
        spanV = self.basis_v.FindSpan(start_knot_v, num_ctrlpts_v, eta, knot_vector_v)
        spanW = self.basis_w.FindSpan(start_knot_w, num_ctrlpts_w, zeta, knot_vector_w)
        NurbsU = self.basis_u.SecondDersBasisFuncs(start_knot_u, spanU, xi, knot_vector_u)
        NurbsV = self.basis_v.SecondDersBasisFuncs(start_knot_v, spanV, eta, knot_vector_v)
        NurbsW = self.basis_w.SecondDersBasisFuncs(start_knot_w, spanW, zeta, knot_vector_w)

        w     = 0.0
        dwdxi, dwdet, dwdze = 0.0, 0.0, 0.0
        d2wdxi, d2wdet, d2wdze = 0.0, 0.0, 0.0
        d2wdxe, d2wdez, d2wdxz = 0.0, 0.0, 0.0
        uind = spanU - self.basis_u.degree
        for i in ti.static(range(self.basis_w.degree + 1)):
            wind = spanW - self.basis_w.degree + i
            for j in ti.static(range(self.basis_v.degree + 1)):
                vind = spanV - self.basis_v.degree + j
                for k in ti.static(range(self.basis_u.degree + 1)):
                    linear_id = uind + k + vind * num_ctrlpts_u + wind * num_ctrlpts_u * num_ctrlpts_v + start_ctrlpt
                    wgt = weight[linear_id]

                    w += NurbsU[0, k] * NurbsV[0, j] * NurbsW[0, i] * wgt
                    dwdxi += NurbsU[1, k] * NurbsV[0, j] * NurbsW[0, i] * wgt
                    dwdet += NurbsU[0, k] * NurbsV[1, j] * NurbsW[0, i] * wgt
                    dwdze += NurbsU[0, k] * NurbsV[0, j] * NurbsW[1, i] * wgt
                    d2wdxi += NurbsU[2, k] * NurbsV[0, j] * NurbsW[0, i] * wgt
                    d2wdet += NurbsU[0, k] * NurbsV[2, j] * NurbsW[0, i] * wgt
                    d2wdze += NurbsU[0, k] * NurbsV[0, j] * NurbsW[2, i] * wgt
                    d2wdxe += NurbsU[1, k] * NurbsV[1, j] * wgt
                    d2wdez += NurbsV[1, j] * NurbsW[1, i] * wgt
                    d2wdxz += NurbsU[1, k] * NurbsW[1, i] * wgt

        position, dirsU, dirsV, dirsW = ti.Vector.zero(float, self.dimension), ti.Vector.zero(float, self.dimension), ti.Vector.zero(float, self.dimension), ti.Vector.zero(float, self.dimension)
        ddirsUU, ddirsVV, ddirsWW = ti.Vector.zero(float, self.dimension), ti.Vector.zero(float, self.dimension), ti.Vector.zero(float, self.dimension)
        ddirsUV, ddirsVW, ddirsUW = ti.Vector.zero(float, self.dimension), ti.Vector.zero(float, self.dimension), ti.Vector.zero(float, self.dimension)
        # dirsU -> dNdxi, dirsV -> dNdeta, dirs[2] -> dNdzeta
        # ddirs[0] -> d2Ndxi2, ddirs[1] -> d2Ndeta2, ddirs[2] -> d2Ndzeta, ddirs[3] -> d2Ndxideta, ddirs[4] -> d2Ndetadzeta, ddirs[5] -> d2Ndxidzeta
        inv_w = 1 / w
        inv_w2 = 1 / w / w
        inv_w3 = 1 / w / w / w
        uind = spanU - self.basis_u.degree
        for i in ti.static(range(self.basis_w.degree + 1)):
            wind = spanW - self.basis_w.degree + i
            for j in ti.static(range(self.basis_v.degree + 1)):
                vind = spanV - self.basis_v.degree + j
                for k in ti.static(range(self.basis_u.degree + 1)):
                    linear_id = uind + k + vind * num_ctrlpts_u + wind * num_ctrlpts_u * num_ctrlpts_v + start_ctrlpt
                    wgt = weight[linear_id]
                    fac = wgt / (w * w)
                    nmp = NurbsU[0, k] * NurbsV[0, j] * NurbsW[0, i]

                    position += nmp * fac * w * point[linear_id]
                    dirsU += (NurbsU[1, k] * NurbsV[0, j] * NurbsW[0, i] * w - nmp * dwdxi) * fac * point[linear_id]
                    dirsV += (NurbsU[0, k] * NurbsV[1, j] * NurbsW[0, i] * w - nmp * dwdet) * fac * point[linear_id]
                    dirsW += (NurbsU[0, k] * NurbsV[0, j] * NurbsW[1, i] * w - nmp * dwdze) * fac * point[linear_id]
                    ddirsUU += (NurbsU[2, k] * NurbsV[0, j] * NurbsW[0, i] * inv_w - 2 * NurbsU[1, k] * NurbsV[0, j] * NurbsW[0, i] * dwdxi * inv_w2 - NurbsU[0, k] * NurbsV[0, j] * NurbsW[0, i] * d2wdxi * inv_w2 + 2 * NurbsU[0, k] * NurbsV[0, j] * NurbsW[0, i] * dwdxi * dwdxi * inv_w3) * wgt * point[linear_id]
                    ddirsVV += (NurbsU[0, k] * NurbsV[2, j] * NurbsW[0, i] * inv_w - 2 * NurbsU[0, k] * NurbsV[1, j] * NurbsW[0, i] * dwdet * inv_w2 - NurbsU[0, k] * NurbsV[0, j] * NurbsW[0, i] * d2wdet * inv_w2 + 2 * NurbsU[0, k] * NurbsV[0, j] * NurbsW[0, i] * dwdet * dwdet * inv_w3) * wgt * point[linear_id]
                    ddirsWW += (NurbsU[0, k] * NurbsV[0, j] * NurbsW[2, i] * inv_w - 2 * NurbsU[0, k] * NurbsV[0, j] * NurbsW[1, i] * dwdze * inv_w2 - NurbsU[0, k] * NurbsV[0, j] * NurbsW[0, i] * d2wdze * inv_w2 + 2 * NurbsU[0, k] * NurbsV[0, j] * NurbsW[0, i] * dwdze * dwdze * inv_w3) * wgt * point[linear_id]
                    ddirsUV += (NurbsU[1, k]*NurbsV[1, j] * inv_w - NurbsU[1, k] * NurbsV[0, j] * dwdet * inv_w2 - NurbsU[0, k] * NurbsV[1, j] * dwdxi * inv_w2 - NurbsU[0, k] * NurbsV[0, j] * d2wdxe * inv_w2 + 2 * NurbsU[0, k] * NurbsV[0, j] * dwdxi * dwdet * inv_w3) * wgt * point[linear_id]
                    ddirsVW += (NurbsV[1, j]*NurbsW[1, i] * inv_w - NurbsV[1, j] * NurbsW[0, i] * dwdxi * inv_w2 - NurbsV[0, j] * NurbsW[1, i] * dwdet * inv_w2 - NurbsV[0, j] * NurbsW[0, i] * d2wdez * inv_w2 + 2 * NurbsV[0, j] * NurbsW[0, i] * dwdet * dwdze * inv_w3) * wgt * point[linear_id]
                    ddirsUW += (NurbsU[1, k]*NurbsW[1, i] * inv_w - NurbsW[1, i] * NurbsU[0, k] * dwdet * inv_w2 - NurbsW[0, i] * NurbsU[1, k] * dwdze * inv_w2 - NurbsU[0, k] * NurbsW[0, i] * d2wdxz * inv_w2 + 2 * NurbsU[0, k] * NurbsW[0, i] * dwdxi * dwdze * inv_w3) * wgt * point[linear_id]
        return position, dirsU, dirsV, dirsW, ddirsUU, ddirsVV, ddirsWW, ddirsUV, ddirsVW, ddirsUW


@ti.func
def get_distance_to_curve(start_knot, start_ctrlpt, num_knot, knot_vector_u, ctrlpts, weight, point, basis):
    num_ctrlpts = num_knot - 1 - basis.basis_u.degree

    u, minDist = 0., 1e15
    for i in range(num_ctrlpts):
        dist2 = Squared(ctrlpts[i + start_ctrlpt] - point)
        if dist2 < minDist:
            u = 0.5 * (knot_vector_u[start_knot + i] + knot_vector_u[start_knot + i + 1])
            minDist = dist2
    
    iter, du, distance, residual1 = 0, 0., 0., ti.Vector.zero(float, point.n)
    while iter < 50:
        u = clamp(knot_vector_u[start_knot + basis.basis_u.degree], knot_vector_u[start_knot + num_ctrlpts], u)
        position, dirsU, ddirsUU = basis.NurbsBasisInterpolations2ndDers1d(start_knot, start_ctrlpt, num_knot, u, knot_vector_u, ctrlpts, weight)
        residual1 = position - point
        if Squared(residual1) < TOL:
            distance = 0.
            break
        
        residual2 = residual1.dot(dirsU) / (residual1.norm() * dirsU.norm())
        if abs(residual2) < TOL:
            distance = residual1.norm()
            break
        
        f = residual1.dot(dirsU)
        det = Squared(dirsU) + residual1.dot(ddirsUU)
        du = f / det
        r4 = du * du * Squared(dirsU)
        if r4 < TOL: 
            distance = residual1.norm()
            break
        u -= du
        iter += 1
    return u, distance, residual1

@ti.func
def get_distance_to_surface(start_knot_u, start_knot_v, start_ctrlpt, num_knot_u, num_knot_v, knot_vector_u, knot_vector_v, ctrlpts, weight, point, basis):
    num_ctrlpts_u = num_knot_u - basis.basis_u.degree - 1
    num_ctrlpts_v = num_knot_v - basis.basis_v.degree - 1

    u, v, minDist = 0., 0., 1e15
    for j in range(num_ctrlpts_v):
        for i in range(num_ctrlpts_u):
            linear_id = i + j * num_ctrlpts_u
            dist2 = Squared(ctrlpts[linear_id + start_ctrlpt] - point)
            if dist2 < minDist:
                u = 0.5 * (knot_vector_u[start_knot_u + i] + knot_vector_u[start_knot_u + i + 1 + basis.basis_u.degree])
                v = 0.5 * (knot_vector_v[start_knot_v + j] + knot_vector_v[start_knot_v + j + 1 + basis.basis_v.degree])
                minDist = dist2
    
    iter, du, dv, distance, residual1 = 0, 0., 0., 0., ti.Vector.zero(float, point.n)
    while iter < 50:
        position, dirsU, dirsV, ddirsUU, ddirsVV, ddirsUV = basis.NurbsBasisInterpolations2ndDers2d(start_knot_u, start_knot_v, start_ctrlpt, num_knot_u, num_knot_v, u, v, knot_vector_u, knot_vector_v, ctrlpts, weight)
        residual1 = position - point
        if Squared(residual1) < TOL:
            distance = 0.
            break
        
        residual2 = residual1.dot(dirsU) / (residual1.norm() * dirsU.norm())
        residual3 = residual1.dot(dirsV) / (residual1.norm() * dirsV.norm())
        if abs(residual2) < TOL and abs(residual3) < TOL:
            distance = residual1.norm()
            break
        
        f = residual1.dot(dirsU)
        g = residual1.dot(dirsV)
        a = Squared(dirsU) + residual1.dot(ddirsUU)
        b = dirsU.dot(dirsV) + residual1.dot(ddirsUV)
        d = Squared(dirsV) + residual1.dot(ddirsVV)
        inv_jac = inverse_matrix_2x2(mat2x2([a, b], [b, d]))
        delta = inv_jac @ -vec2f(f, g)
        du, dv = delta[0], delta[1]

        u = clamp(knot_vector_u[start_knot_u + basis.basis_u.degree], knot_vector_u[start_knot_u + num_ctrlpts_u], u + du)
        v = clamp(knot_vector_v[start_knot_v + basis.basis_v.degree], knot_vector_v[start_knot_v + num_ctrlpts_v], v + dv)

        r4 = du * du * Squared(dirsU) + dv * dv * Squared(dirsV)
        if r4 < TOL: 
            distance = residual1.norm()
            break
        iter += 1
    if iter == 50:
        distance = residual1.norm()
    return u, v, distance, residual1
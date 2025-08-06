import pytest

from shapely.errors import ShapelyDeprecationWarning
from shapely.geometry import Polygon, MultiPolygon, asMultiPolygon
from shapely.geometry.base import dump_coords

from . import unittest, numpy, test_int_types, shapely20_deprecated
from .test_multi import MultiGeometryTestCase


class MultiPolygonTestCase(MultiGeometryTestCase):

    def test_multipolygon(self):

        # From coordinate tuples
        geom = MultiPolygon(
            [(((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)),
              [((0.25, 0.25), (0.25, 0.5), (0.5, 0.5), (0.5, 0.25))])])
        self.assertIsInstance(geom, MultiPolygon)
        self.assertEqual(len(geom.geoms), 1)
        self.assertEqual(
            dump_coords(geom),
            [[(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0),
              [(0.25, 0.25), (0.25, 0.5), (0.5, 0.5), (0.5, 0.25),
               (0.25, 0.25)]]])

        # Or from polygons
        p = Polygon(((0, 0), (0, 1), (1, 1), (1, 0)),
                    [((0.25, 0.25), (0.25, 0.5), (0.5, 0.5), (0.5, 0.25))])
        geom = MultiPolygon([p])
        self.assertEqual(len(geom.geoms), 1)
        self.assertEqual(
            dump_coords(geom),
            [[(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0),
              [(0.25, 0.25), (0.25, 0.5), (0.5, 0.5), (0.5, 0.25),
               (0.25, 0.25)]]])

        # Or from another multi-polygon
        geom2 = MultiPolygon(geom)
        self.assertEqual(len(geom2.geoms), 1)
        self.assertEqual(
            dump_coords(geom2),
            [[(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0),
              [(0.25, 0.25), (0.25, 0.5), (0.5, 0.5), (0.5, 0.25),
               (0.25, 0.25)]]])

        # Sub-geometry Access
        self.assertIsInstance(geom.geoms[0], Polygon)
        self.assertEqual(
            dump_coords(geom.geoms[0]),
            [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0),
             [(0.25, 0.25), (0.25, 0.5), (0.5, 0.5), (0.5, 0.25),
              (0.25, 0.25)]])
        with self.assertRaises(IndexError):  # index out of range
            geom.geoms[1]

        # Geo interface
        self.assertEqual(
            geom.__geo_interface__,
            {'type': 'MultiPolygon',
             'coordinates': [(((0.0, 0.0), (0.0, 1.0), (1.0, 1.0),
                               (1.0, 0.0), (0.0, 0.0)),
                              ((0.25, 0.25), (0.25, 0.5), (0.5, 0.5),
                               (0.5, 0.25), (0.25, 0.25)))]})

    @shapely20_deprecated
    def test_multipolygon_adapter(self):
        # Adapter
        coords = ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0))
        holes_coords = [((0.25, 0.25), (0.25, 0.5), (0.5, 0.5), (0.5, 0.25))]
        mpa = asMultiPolygon([(coords, holes_coords)])
        self.assertEqual(len(mpa.geoms), 1)
        self.assertEqual(len(mpa.geoms[0].exterior.coords), 5)
        self.assertEqual(len(mpa.geoms[0].interiors), 1)
        self.assertEqual(len(mpa.geoms[0].interiors[0].coords), 5)

    @shapely20_deprecated
    def test_subgeom_access(self):
        poly0 = Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)])
        poly1 = Polygon([(0.25, 0.25), (0.25, 0.5), (0.5, 0.5), (0.5, 0.25)])
        self.subgeom_access_test(MultiPolygon, [poly0, poly1])


def test_fail_list_of_multipolygons():
    """A list of multipolygons is not a valid multipolygon ctor argument"""
    multi = MultiPolygon([(((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)), [((0.25, 0.25), (0.25, 0.5), (0.5, 0.5), (0.5, 0.25))])])
    with pytest.raises(ValueError):
        MultiPolygon([multi])


def test_multipolygon_adapter_deprecated():
    coords = ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0))
    holes_coords = [((0.25, 0.25), (0.25, 0.5), (0.5, 0.5), (0.5, 0.25))]
    with pytest.warns(ShapelyDeprecationWarning, match="proxy geometries"):
        asMultiPolygon([(coords, holes_coords)])


def test_iteration_deprecated():
    geom = MultiPolygon(
        [(((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)),
          [((0.25, 0.25), (0.25, 0.5), (0.5, 0.5), (0.5, 0.25))])])
    with pytest.warns(ShapelyDeprecationWarning, match="Iteration"):
        for g in geom:
            pass


def test_getitem_deprecated():
    geom = MultiPolygon(
        [(((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)),
          [((0.25, 0.25), (0.25, 0.5), (0.5, 0.5), (0.5, 0.25))])])
    with pytest.warns(ShapelyDeprecationWarning, match="__getitem__"):
        part = geom[0]


@shapely20_deprecated
@pytest.mark.filterwarnings("error:An exception was ignored")  # NumPy 1.21
def test_numpy_object_array():
    np = pytest.importorskip("numpy")

    geom = MultiPolygon(
        [(((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)),
          [((0.25, 0.25), (0.25, 0.5), (0.5, 0.5), (0.5, 0.25))])])
    ar = np.empty(1, object)
    ar[:] = [geom]
    assert ar[0] == geom

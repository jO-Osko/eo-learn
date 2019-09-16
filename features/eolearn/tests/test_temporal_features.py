"""
Credits:
Copyright (c) 2017-2019 Matej Aleksandrov, Matej Batič, Andrej Burja, Eva Erzin (Sinergise)
Copyright (c) 2017-2019 Grega Milčinski, Matic Lubej, Devis Peresutti, Jernej Puc, Tomislav Slijepčević (Sinergise)
Copyright (c) 2017-2019 Blaž Sovdat, Jovan Višnjić, Anže Zupanc, Lojze Žust (Sinergise)

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

import os.path
import unittest
import numpy as np

from datetime import date, timedelta

from eolearn.core import EOPatch, FeatureType
from eolearn.features import AddMaxMinNDVISlopeIndicesTask, AddMaxMinTemporalIndicesTask, \
    AddSpatioTemporalFeaturesTask, AddStreamTemporalFeaturesTask


def perdelta(start, end, delta):
    curr = start
    while curr < end:
        yield curr
        curr += delta


class TestTemporalFeaturesTasks(unittest.TestCase):
    TEST_PATCH_FILENAME = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'TestInputs', 'TestPatch')

    def test_temporal_indices(self):
        """ Test case for computation of argmax/argmin of NDVI and another band

        Cases with and without data masking are tested
        """
        # EOPatch
        eopatch = EOPatch()
        t, h, w, c = 5, 3, 3, 2
        # NDVI
        ndvi_shape = (t, h, w, 1)
        # VAlid data mask
        valid_data = np.ones(ndvi_shape, np.bool)
        valid_data[0] = 0
        valid_data[-1] = 0
        # Fill in eopatch
        eopatch.add_feature(FeatureType.DATA, 'NDVI', np.arange(np.prod(ndvi_shape)).reshape(ndvi_shape))
        eopatch.add_feature(FeatureType.MASK, 'IS_DATA', np.ones(ndvi_shape, dtype=np.int16))
        eopatch.add_feature(FeatureType.MASK, 'VALID_DATA', valid_data)
        # Task
        add_ndvi = AddMaxMinTemporalIndicesTask(mask_data=False)
        # Run task
        new_eopatch = add_ndvi(eopatch)
        # Asserts
        self.assertTrue(np.array_equal(new_eopatch.data_timeless['ARGMIN_NDVI'], np.zeros((h, w, 1))))
        self.assertTrue(np.array_equal(new_eopatch.data_timeless['ARGMAX_NDVI'], (t-1)*np.ones((h, w, 1))))
        del add_ndvi, new_eopatch
        # Repeat with valid dat amask
        add_ndvi = AddMaxMinTemporalIndicesTask(mask_data=True)
        new_eopatch = add_ndvi(eopatch)
        # Asserts
        self.assertTrue(np.array_equal(new_eopatch.data_timeless['ARGMIN_NDVI'], np.ones((h, w, 1))))
        self.assertTrue(np.array_equal(new_eopatch.data_timeless['ARGMAX_NDVI'], (t-2)*np.ones((h, w, 1))))
        del add_ndvi, new_eopatch, valid_data
        # BANDS
        bands_shape = (t, h, w, c)
        eopatch.add_feature(FeatureType.DATA, 'BANDS', np.arange(np.prod(bands_shape)).reshape(bands_shape))
        add_bands = AddMaxMinTemporalIndicesTask(data_feature='BANDS',
                                                 data_index=1,
                                                 amax_data_feature='ARGMAX_B1',
                                                 amin_data_feature='ARGMIN_B1',
                                                 mask_data=False)
        new_eopatch = add_bands(eopatch)
        self.assertTrue(np.array_equal(new_eopatch.data_timeless['ARGMIN_B1'], np.zeros((h, w, 1))))
        self.assertTrue(np.array_equal(new_eopatch.data_timeless['ARGMAX_B1'], (t-1)*np.ones((h, w, 1))))

    def test_ndvi_slope_indices(self):
        """ Test case for computation of argmax/argmin of NDVI slope

            The NDVI is a sinusoid over 0-pi over the temporal dimension

            Cases with and without data masking are tested
        """
        # Slope needs timestamps
        timestamp = perdelta(date(2018, 3, 1), date(2018, 3, 11), timedelta(days=1))
        # EOPatch
        eopatch = EOPatch(timestamp=list(timestamp))
        t, h, w, = 10, 3, 3
        # NDVI is a sinusoid where max slope is at index 1 and min slope at index 8
        ndvi_shape = (t, h, w, 1)
        xx = np.zeros(ndvi_shape, np.float32)
        x = np.linspace(0, np.pi, t)
        xx[:, :, :, :] = x[:, None, None, None]
        # Valid data mask
        valid_data = np.ones(ndvi_shape, np.uint8)
        valid_data[1] = 0
        valid_data[-1] = 0
        valid_data[4] = 0
        # Fill EOPatch
        eopatch.add_feature(FeatureType.DATA, 'NDVI', np.sin(xx))
        eopatch.add_feature(FeatureType.MASK, 'IS_DATA', np.ones(ndvi_shape, np.bool))
        eopatch.add_feature(FeatureType.MASK, 'VALID_DATA', valid_data)
        # Tasks
        add_ndvi = AddMaxMinTemporalIndicesTask(mask_data=False)
        add_ndvi_slope = AddMaxMinNDVISlopeIndicesTask(mask_data=False)
        # Run
        new_eopatch = add_ndvi_slope(add_ndvi(eopatch))
        # Assert
        self.assertTrue(np.array_equal(new_eopatch.data_timeless['ARGMIN_NDVI'], (t-1)*np.ones((h, w, 1))))
        self.assertTrue(np.array_equal(new_eopatch.data_timeless['ARGMAX_NDVI'], (t//2-1)*np.ones((h, w, 1))))
        self.assertTrue(np.array_equal(new_eopatch.data_timeless['ARGMIN_NDVI_SLOPE'], (t-2)*np.ones((h, w, 1))))
        self.assertTrue(np.array_equal(new_eopatch.data_timeless['ARGMAX_NDVI_SLOPE'], np.ones((h, w, 1))))
        del add_ndvi_slope, add_ndvi, new_eopatch
        # Run on valid data only now
        add_ndvi = AddMaxMinTemporalIndicesTask(mask_data=True)
        add_ndvi_slope = AddMaxMinNDVISlopeIndicesTask(mask_data=True)
        # Run
        new_eopatch = add_ndvi_slope(add_ndvi(eopatch))
        # Assert
        self.assertTrue(np.array_equal(new_eopatch.data_timeless['ARGMIN_NDVI'], 0 * np.ones((h, w, 1))))
        self.assertTrue(np.array_equal(new_eopatch.data_timeless['ARGMAX_NDVI'], (t // 2) * np.ones((h, w, 1))))
        self.assertTrue(np.array_equal(new_eopatch.data_timeless['ARGMIN_NDVI_SLOPE'], (t - 3) * np.ones((h, w, 1))))
        self.assertTrue(np.array_equal(new_eopatch.data_timeless['ARGMAX_NDVI_SLOPE'], 2 * np.ones((h, w, 1))))

    def test_stf_task(self):
        """ Test case for computation of spatio-temporal features

            The NDVI is a sinusoid over 0-pi over the temporal dimension, while bands is an array with values equal to
            the temporal index
        """
        # Timestamps
        timestamp = perdelta(date(2018, 3, 1), date(2018, 3, 11), timedelta(days=1))
        # EOPatch
        eopatch = EOPatch(timestamp=list(timestamp))
        # Shape of arrays
        t, h, w, c = 10, 3, 3, 2
        # NDVI is a sinusoid where max slope is at index 1 and min slope at index 8
        ndvi_shape = (t, h, w, 1)
        bands_shape = (t, h, w, c)
        xx = np.zeros(ndvi_shape, np.float32)
        x = np.linspace(0, np.pi, t)
        xx[:, :, :, :] = x[:, None, None, None]
        # Bands are arrays with values equal to the temporal index
        bands = np.ones(bands_shape)*np.arange(t)[:, None, None, None]
        # Add features to eopatch
        eopatch.add_feature(FeatureType.DATA, 'NDVI', np.sin(xx))
        eopatch.add_feature(FeatureType.DATA, 'BANDS', bands)
        eopatch.add_feature(FeatureType.MASK, 'IS_DATA', np.ones(ndvi_shape, np.bool))
        # Tasks
        add_ndvi = AddMaxMinTemporalIndicesTask(mask_data=False)
        add_bands = AddMaxMinTemporalIndicesTask(data_feature='BANDS',
                                                 data_index=1,
                                                 amax_data_feature='ARGMAX_B1',
                                                 amin_data_feature='ARGMIN_B1',
                                                 mask_data=False)
        add_ndvi_slope = AddMaxMinNDVISlopeIndicesTask(mask_data=False)
        add_stf = AddSpatioTemporalFeaturesTask(argmax_red='ARGMAX_B1', data_feature='BANDS', indices=[0, 1])
        # Run tasks
        new_eopatch = add_stf(add_ndvi_slope(add_bands(add_ndvi(eopatch))))
        # Asserts
        self.assertTrue(new_eopatch.data_timeless['STF'].shape == (h, w, c*5))
        self.assertTrue(np.array_equal(new_eopatch.data_timeless['STF'][:, :, 0:c], 4*np.ones((h, w, c))))
        self.assertTrue(np.array_equal(new_eopatch.data_timeless['STF'][:, :, c:2*c], 9*np.ones((h, w, c))))
        self.assertTrue(np.array_equal(new_eopatch.data_timeless['STF'][:, :, 2*c:3*c], np.ones((h, w, c))))
        self.assertTrue(np.array_equal(new_eopatch.data_timeless['STF'][:, :, 3*c:4*c], 8*np.ones((h, w, c))))
        self.assertTrue(np.array_equal(new_eopatch.data_timeless['STF'][:, :, 4*c:5*c], 9*np.ones((h, w, c))))

    def test_streamtemporalfeatures_task(self):
        patch = EOPatch.load(self.TEST_PATCH_FILENAME)

        task = AddStreamTemporalFeaturesTask(data_feature="ndvi", ndvi_feature_name="ndvi")

        task.execute(patch)

        dates = perdelta(date(2018, 3, 1), date(2018, 3, 21), timedelta(days=1))
        patch = EOPatch(timestamp=list(dates))
        ndvi = np.array([1, 2, 4, -1, -2, 3, 5, 7, 9, 10, 12, 11, 12, 12, 11, 9, 7, 3, -1, 2]).reshape(-1, 1, 1, 1)
        names = ['NDVI_max_val', 'NDVI_min_val', 'NDVI_mean_val', 'NDVI_sd_val', 'NDVI_diff_max', 'NDVI_diff_min',
                 'NDVI_diff_diff', 'NDVI_max_mean_feature', 'NDVI_max_mean_len', 'NDVI_max_mean_surf', 'NDVI_pos_len',
                 'NDVI_pos_surf', 'NDVI_pos_rate', 'NDVI_pos_tran', 'NDVI_neg_len', 'NDVI_neg_surf', 'NDVI_neg_rate',
                 'NDVI_neg_tran']
        values = np.array(
            [12., -2., 5.8, 4.66476152, 5., 0., 5., 12., 12., 120., 8., 70.,
             1.75, 1., 5., 40.5, -2.6, 1.])

        patch.add_feature(FeatureType.DATA, 'NDVI', ndvi)
        task = AddStreamTemporalFeaturesTask(data_feature="NDVI",
                                             ndvi_feature_name="NDVI")

        task.execute(patch)

        result = task.get_data(patch)

        self.assertEqual(result[0], names)
        self.assertTrue(np.allclose(result[1], values))

if __name__ == '__main__':
    unittest.main()

import unittest
import numpy as np

from datetime import date, timedelta

from eolearn.core import EOPatch, FeatureType
from eolearn.features import AddMaxMinNDVISlopeIndicesTask, AddMaxMinTemporalIndicesTask,\
    AddSpatioTemporalFeaturesTask, AddStreamTemporalFeaturesTask


def perdelta(start, end, delta):
    curr = start
    while curr < end:
        yield curr
        curr += delta


class TestTemporalFeaturesTasks(unittest.TestCase):

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

    def test_stream_temporal_indices(self):
        """ Test case for aggregate temporal indices of custom band

        Tests are preformed on a hand built single pixel image with hand calculated values.
        """
        days_delta = 10

        timestamp = list(perdelta(date(2018, 3, 1), date(2018, 10, 1), timedelta(days=days_delta)))

        # Normalize to [-1,1] (range of usual eo-indices)
        data = np.array([
            [
                [0, 1, 3, 0, -2, 1, 3, 5, 7, 7.8, 8, 8.2, 8.1, 7.9, 5, 3, 1, -3, -0.5, 1],
            ],
        ]) / 10
        h, w, t = data.shape

        data_shape = (t, h, w, 1)
        valid_data = np.ones(data_shape)
        data_name = "NDVI"
        data = data[..., np.newaxis]
        # Fill
        eopatch = EOPatch(timestamp=timestamp)
        eopatch.add_feature(FeatureType.DATA, data_name, data.swapaxes(0, 2).swapaxes(1, 2))
        eopatch.add_feature(FeatureType.MASK, 'IS_DATA', valid_data)
        eopatch.add_feature(FeatureType.MASK, 'VALID_DATA', valid_data)

        add_features = AddStreamTemporalFeaturesTask(data_feature=data_name, mask_data=True, interval_tolerance=0.1,
                                                     window_size=2)
        new_eopatch = add_features(eopatch)

        self.assertTrue(np.allclose(new_eopatch.data_timeless[add_features.max_val_feature], np.array([[[8.2]]])/10))
        self.assertTrue(np.allclose(new_eopatch.data_timeless[add_features.min_val_feature], np.array([[[-3]]])/10))
        self.assertTrue(np.allclose(new_eopatch.data_timeless[add_features.mean_val_feature], np.array([[[3.225]]])/10))
        self.assertTrue(np.allclose(new_eopatch.data_timeless[add_features.sd_val_feature], np.array([[[3.57308]]])/10))

        self.assertTrue(np.allclose(new_eopatch.data_timeless[add_features.diff_max_feature], np.array([[[4]]])/10))
        self.assertTrue(np.allclose(new_eopatch.data_timeless[add_features.diff_min_feature], np.array([[[0.1]]])/10))
        self.assertTrue(np.allclose(new_eopatch.data_timeless[add_features.diff_diff_feature], np.array([[[3.9]]])/10))

        self.assertTrue(np.allclose(new_eopatch.data_timeless[add_features.max_mean_feature], np.array([[[8.15]]])/10))

        # 8.15 * 0.9 ~ 7.4
        self.assertTrue(np.allclose(new_eopatch.data_timeless[add_features.max_mean_len_feature],
                                    np.array([[[days_delta * 4]]])))
        # [7.8, 8, 8.2, 8.1, 7.9]
        self.assertTrue(np.allclose(new_eopatch.data_timeless[add_features.max_mean_surf_feature],
                                    np.array([[[72.15]]])))

        # From -2 to 8.2
        self.assertTrue(np.allclose(new_eopatch.data_timeless[add_features.pos_len_feature],
                                    np.array([[[days_delta * 7]]])))
        self.assertTrue(np.allclose(new_eopatch.data_timeless[add_features.pos_surf_feature],
                                    np.array([[[104.9]]])))
        self.assertTrue(np.allclose(new_eopatch.data_timeless[add_features.pos_rate_feature],
                                    np.array([[[(8.2 - -2)/(days_delta*7*10)]]])))

        self.assertTrue(np.allclose(new_eopatch.data_timeless[add_features.neg_len_feature],
                                    np.array([[[days_delta * 5]]])))

        self.assertTrue(np.allclose(new_eopatch.data_timeless[add_features.neg_surf_feature],
                                    np.array([[[69.45]]])))
        self.assertTrue(np.allclose(new_eopatch.data_timeless[add_features.neg_rate_feature],
                                    np.array([[[(-3 - 8.1)/(days_delta*5*10)]]])))

        self.assertTrue(np.all(new_eopatch.data_timeless[add_features.pos_transition_feature] == np.array([[[1]]])))
        self.assertTrue(np.all(new_eopatch.data_timeless[add_features.neg_transition_feature] == np.array([[[1]]])))

        add_features = AddStreamTemporalFeaturesTask(data_feature=data_name, mask_data=True, interval_tolerance=0.1,
                                                     window_size=3)
        new_eopatch_sl3 = add_features(eopatch)

        # 3, -3
        self.assertTrue(np.allclose(new_eopatch_sl3.data_timeless[add_features.diff_max_feature],
                                    np.array([[[6]]]) / 10))
        # 8.0, 8.2
        self.assertTrue(np.allclose(new_eopatch_sl3.data_timeless[add_features.diff_min_feature],
                                    np.array([[[0.2]]])/10))
        # 5.8
        self.assertTrue(np.allclose(new_eopatch_sl3.data_timeless[add_features.diff_diff_feature],
                                    np.array([[[5.8]]])/10))
        # [8, 8.2, 8.1]
        self.assertTrue(np.allclose(new_eopatch_sl3.data_timeless[add_features.max_mean_feature],
                                    np.array([[[8.1]]])/10))

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


if __name__ == '__main__':
    unittest.main()

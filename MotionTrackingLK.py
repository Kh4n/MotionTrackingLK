import tensorflow as tf
import cv2 as cv
import numpy as np

def gaussian(x, sigma):
    return 1/(sigma*math.sqrt(2*math.pi))*math.e**(-1/2*(x/sigma)**2)

class MotionTrackingLK(tf.keras.layers.Layer):
    def __init__(self, num_tracks, window_pixel_wh=21, sigma=2, iterations=5, **kwargs):
        self.sigma = sigma
        assert(num_tracks > 1)
        assert(window_pixel_wh >= 3)
        self.num_tracks = num_tracks
        self.win_pixel_wh = window_pixel_wh
        self.iterations = iterations
        super(MotionTrackingLK, self).__init__(**kwargs)

    def build(self, input_shape):
        # grab the dimensions of the image here so we can use them later. also will throw errors early for users
        self.h = input_shape[1][1+1]
        self.w = input_shape[1][2+1]
        self.c = input_shape[1][3+1]

        self.scale = self.win_pixel_wh / min(self.w, self.h)

        # we scale to the smaller axis and then apply transforms to that resulting square
        # originally was [0.0, 1.0], but this resulted in the model being unable to learn. not sure why. possibly because tanh learns better than sigmoid
        x_t, y_t = tf.meshgrid(
            tf.linspace(-1.0, 1.0, self.win_pixel_wh),
            tf.linspace(-1.0, 1.0, self.win_pixel_wh),
        )
        self.sampling_grid = tf.stack([
            tf.reshape(x_t, [self.win_pixel_wh*self.win_pixel_wh]),
            tf.reshape(y_t, [self.win_pixel_wh*self.win_pixel_wh]),
        ])

        self.sobel_x = tf.constant([
                [-1.,  0.,  1.],
                [-2.,  0.,  2.],
                [-1.,  0.,  1.],
            ],
            shape=[3, 3, 1, 1]
        )
        self.sobel_y = tf.constant([
                [-1., -2., -1.],
                [ 0.,  0.,  0.],
                [ 1.,  2.,  1.],
            ],
            shape=[3, 3, 1, 1]
        )

        self.scharr_x = tf.constant([
                [-3.,   0.,  3.],
                [-10.,  0.,  10.],
                [-3.,   0.,  3.],
            ],
            shape=[3, 3, 1, 1]
        )
        self.scharr_y = tf.constant([
                [-3., -10., -3.],
                [ 0.,   0.,  0.],
                [ 3.,  10.,  3.],
            ],
            shape=[3, 3, 1, 1]
        )

        # weights = np.ones([self.win_pixel_wh, self.win_pixel_wh])
        weights = np.empty([window_pixel_wh, window_pixel_wh])
        center = window_pixel_wh//2
        for y in range(window_pixel_wh):
            for x in range(window_pixel_wh):
                weights[y, x] = (x-center)**2 + (y-center)**2

        weights = gaussian(np.sqrt(weights), self.sigma)
        self.win_weights = tf.constant(weights, shape=[1, 1, self.win_pixel_wh, self.win_pixel_wh, 1], dtype=tf.float32)
        # print(weights)
        # tf.print(weights)
        # tf.print(tf.reduce_max(weights))

        super(MotionTrackingLK, self).build(input_shape)
    
    def sample_ntracks_from_2frames(self, samples, frames):
        x = ((samples[:, :, 0]) * self.win_pixel_wh) * 0.5
        y = ((samples[:, :, 1]) * self.win_pixel_wh) * 0.5

        x = tf.reshape(tf.tile(tf.expand_dims(x, axis=1), [1, 2, 1, 1]), [-1, self.num_tracks*self.win_pixel_wh**2])
        y = tf.reshape(tf.tile(tf.expand_dims(y, axis=1), [1, 2, 1, 1]), [-1, self.num_tracks*self.win_pixel_wh**2])

        x0 = tf.floor(x)
        x1 = x0 + 1
        y0 = tf.floor(y)
        y1 = y0 + 1

        x0 = tf.clip_by_value(x0, 0, self.w-1)
        x1 = tf.clip_by_value(x1, 0, self.w-1)
        y0 = tf.clip_by_value(y0, 0, self.h-1)
        y1 = tf.clip_by_value(y1, 0, self.h-1)

        wa = tf.expand_dims((y1-y) * (x1-x), axis=-1)
        wb = tf.expand_dims((y1-y) * (x-x0), axis=-1)
        wc = tf.expand_dims((y-y0) * (x1-x), axis=-1)
        wd = tf.expand_dims((y-y0) * (x-x0), axis=-1)

        x0 = tf.cast(x0, tf.int32)
        x1 = tf.cast(x1, tf.int32)
        y0 = tf.cast(y0, tf.int32)
        y1 = tf.cast(y1, tf.int32)

        # necessary to that first dimension is equal. makes it so that we are repeatedly sampling for each image
        tiled_imgs = tf.reshape(frames, [-1, self.h, self.w, self.c])

        # batch dimension in this case goes through first frame for each batch, then second frame
        Ia = tf.gather_nd(tiled_imgs, tf.stack([y0, x0], axis=-1), batch_dims=1)
        Ib = tf.gather_nd(tiled_imgs, tf.stack([y0, x1], axis=-1), batch_dims=1)
        Ic = tf.gather_nd(tiled_imgs, tf.stack([y1, x0], axis=-1), batch_dims=1)
        Id = tf.gather_nd(tiled_imgs, tf.stack([y1, x1], axis=-1), batch_dims=1)

        # a = tf.stack([y0, x0], axis=-1)
        # print(tf.shape(y0))
        # print(tf.shape(a))
        # print(tf.shape(wa))
        # print(tf.shape(Ia))
        # print(tf.shape(tiled_imgs))

        return tf.reshape(wa*Ia + wb*Ib + wc*Ic + wd*Id, [-1, 2, self.num_tracks, self.win_pixel_wh, self.win_pixel_wh, self.c])
  
    def calc_velocity_2frames_ntracks_LK(self, first_frame, second_frame):
        ff_comb = tf.reshape(first_frame, [-1, self.win_pixel_wh, self.win_pixel_wh, self.c])

        Ix = tf.reshape(
            tf.nn.convolution(ff_comb, self.sobel_x, padding="SAME"),
            [-1, self.num_tracks, self.win_pixel_wh, self.win_pixel_wh, self.c]
        )
        sum_Ix2 = tf.reduce_sum(Ix*Ix*self.win_weights, axis=[2,3,4])

        Iy = tf.reshape(
            tf.nn.convolution(ff_comb, self.sobel_y, padding="SAME"),
            [-1, self.num_tracks, self.win_pixel_wh, self.win_pixel_wh, self.c]
        )
        sum_Iy2 = tf.reduce_sum(Iy*Iy*self.win_weights, axis=[2,3,4])

        sum_IxIy = tf.reduce_sum(Ix*Iy*self.win_weights, axis=[2,3,4])
        ATA = tf.reshape(tf.stack([sum_Iy2, -sum_IxIy, -sum_IxIy, sum_Ix2], axis=-1), [-1, self.num_tracks, 2,2])
        ATA_det = tf.reshape(1.0/(sum_Ix2*sum_Iy2 - sum_IxIy*sum_IxIy + 1e-07) , [-1, self.num_tracks, 1,1])
        ATA_1 = ATA_det*ATA

        It = second_frame-first_frame
        sum_IxIt = tf.reduce_sum(Ix*It*self.win_weights, axis=[2,3,4])
        sum_IyIt = tf.reduce_sum(Iy*It*self.win_weights, axis=[2,3,4])
        b = tf.reshape(tf.stack([-sum_IxIt, -sum_IyIt], axis=-1), [-1, self.num_tracks, 2,1])

        VxVy = tf.matmul(ATA_1, b)

        # print(tf.shape(Ix))
        # print(tf.shape(sum_Ix2))
        # print(tf.shape(Iy))
        # print(tf.shape(sum_Iy2))
        # print(tf.shape(It))
        # print(tf.shape(ATA_1))
        # print(tf.shape(b))
        # print(tf.shape(VxVy))
        # tf.print(sum_Ix2, summarize=-1)
        # tf.print(sum_Iy2, summarize=-1)
        # tf.print(sum_IxIy, summarize=-1)
        # tf.print(ATA, summarize=-1)
        # tf.print(ATA_det, summarize=-1)
        # tf.print(ATA_1, summarize=-1)

        return VxVy

    def iterative_LK(self, samples, frames, iterations):
        out = self.sample_ntracks_from_2frames(samples, frames)
        first_frame = out[:, 0]
        factor = 2

        VxVy = self.calc_velocity_2frames_ntracks_LK(first_frame, out[:, 1])
        samples += VxVy
        sum_VxVy = VxVy

        i = tf.constant(1)
        cond = lambda i, s, f, sf, svv, pvv, mm: tf.less(i, iterations)

        def iterate(i, samples, frames, first_frame, sum_VxVy, prev_VxVy, momentum):
            out = self.sample_ntracks_from_2frames(samples, frames)

            VxVy = self.calc_velocity_2frames_ntracks_LK(first_frame, out[:, 1])

            samples += VxVy 
            i += 1
            sum_VxVy += VxVy
            return i, samples, frames, first_frame, sum_VxVy, VxVy, momentum

        _, samples, _, _, sum_VxVy, _, _ = tf.while_loop(cond, iterate, [i, samples, frames, first_frame, sum_VxVy, VxVy, VxVy])
        return samples, sum_VxVy


    def call(self, inputs):
        init_track_locs = tf.reshape(inputs[0], [-1, self.num_tracks, 2, 1]) * 1.0/self.scale
        imgs = inputs[1]
        seq_len = tf.shape(imgs)[1]

        samples = tf.reshape(self.sampling_grid, [1, 1, 2, -1]) + init_track_locs
        first_frame = self.sample_ntracks_from_2frames(samples, imgs[:, 0:2])[:, 0]
        samples, tot_VxVy = self.iterative_LK(samples, imgs[:, 0:2], self.iterations)
        tot_VxVy = tf.concat([tf.reshape(inputs[0], [-1, self.num_tracks, 2, 1]), tot_VxVy], axis=2)

        i = tf.constant(1)
        cond = lambda i, s, imgs, tot_VxVy: tf.less(i, seq_len-1)
        def iterate(i, samples, imgs, tot_VxVy):
            samples, sum_VxVy = self.iterative_LK(samples, imgs[:, i:i+2], self.iterations)
            tot_VxVy = tf.concat([tot_VxVy, sum_VxVy], axis=2)
            i += 1
            return i, samples, imgs, tot_VxVy
        
        _, samples, _, tot_VxVy = tf.while_loop(
            cond, iterate, [i, samples, imgs, tot_VxVy],
            shape_invariants=[i.get_shape(), samples.get_shape(), imgs.get_shape(), tf.TensorShape([None, self.num_tracks, None, 1])]
        )
        tracked = self.sample_ntracks_from_2frames(samples, imgs[:, seq_len-2:seq_len])[:, 1]
        tot_VxVy = tf.reshape(tot_VxVy, [-1, self.num_tracks, seq_len, 2])
        tot_VxVy.set_shape([None, self.num_tracks, 5, 2])
        tf.print(tot_VxVy)
        return tf.stack([first_frame, tracked], axis=1)
        # return tot_VxVy
  
    def compute_output_shape(self, input_shape):
        seq_len = input_shape[1][1]
        return [None, self.num_tracks, seq_len, 2]
  
    def get_config(self):
        base_config = super(MotionTrackingLK, self).get_config()
        return base_config
  
    @classmethod
    def from_config(cls, config):
        return cls(**config)

if __name__ == "__main__":
    import numpy as np
    import math

    window_pixel_wh = 21
    num_tracks = 3
    sigma = 2
    batches = 1
    iterations = 5
    # some test points to be tracked
    transforms = np.asarray(
        [0.95, 1.15, 
        0.95+0.57, 1.15-0.06,
        0.95+0.90, 1.15-0.85]*batches,
        dtype=np.float32
    )
    imgs = np.asarray([
        [
            np.expand_dims(cv.imread("car_dashcam0.png", cv.IMREAD_GRAYSCALE) / 255, axis=-1),
            np.expand_dims(cv.imread("car_dashcam1.png", cv.IMREAD_GRAYSCALE) / 255, axis=-1),
            np.expand_dims(cv.imread("car_dashcam2.png", cv.IMREAD_GRAYSCALE) / 255, axis=-1),
            np.expand_dims(cv.imread("car_dashcam3.png", cv.IMREAD_GRAYSCALE) / 255, axis=-1),
            np.expand_dims(cv.imread("car_dashcam4.png", cv.IMREAD_GRAYSCALE) / 255, axis=-1),
        ]
    ]*batches).astype(np.float32)

    out = np.float32(MotionTrackingLK(
        num_tracks=num_tracks, window_pixel_wh=window_pixel_wh, sigma=sigma, iterations=iterations)([transforms, imgs]
    ))
    print(out.shape)
    cv.imshow(
        f"should be {num_tracks} zoomed in images",
        np.reshape(out, [-1, window_pixel_wh*num_tracks, window_pixel_wh, 1])[0]
        # out[0]
        # out[0, 0]
    )
    cv.waitKey()
    for i in range(2):
        for j in range(num_tracks):
            cv.imshow(f"frame {i}, track {j}", out[0, i,j])
            cv.waitKey()

    with open("OUT", "w") as f:
        with np.printoptions(threshold=np.inf):
            f.write(str(out))
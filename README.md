OpenCV provides accurate methods for affine image stitching. However, functions like cv2.estimateAffinePartial2D() are slow for large images and do not work for very large images (>1GB).

When stitiching together microscopy images into a panorama, the images fit together without rotational or deformational changes. This repo provides tools to aid in quickly stitching together large numbers of microscopy images (e.g., to construct a virtual slide after scanning).


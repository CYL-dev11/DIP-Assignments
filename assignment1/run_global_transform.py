# SA25001037

import cv2
import numpy as np
import gradio as gr

# Global variables for storing source and target control points
points_src = []
points_dst = []
image = None

# Reset control points when a new image is uploaded
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()
    points_dst.clear()
    image = img
    return img

# Record clicked points and visualize them on the image
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]

    # Alternate clicks between source and target points
    if len(points_src) == len(points_dst):
        points_src.append([x, y])
    else:
        points_dst.append([x, y])

    # Draw points (blue: source, red: target) and arrows on the image
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 3, (255, 0, 0), -1)  # Blue for source
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 3, (0, 0, 255), -1)  # Red for target

    # Draw arrows from source to target points
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 2)

    return marked_image

# Point-guided image deformation using Rigid MLS
def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """
    Deform image using Moving Least Squares (Rigid transformation)
    
    Parameters:
        image: input image (H, W, C) as numpy array
        source_pts: source control points (N, 2) from user clicks
        target_pts: target control points (N, 2) from user clicks
        alpha: weight parameter, default 1.0
        eps: small epsilon to avoid division by zero
    
    Returns:
        warped_image: deformed image (same size as input)
    """
    h, w = image.shape[:2]
    n_pts = len(source_pts)
    
    # Convert to float arrays for computation
    src = np.array(source_pts, dtype=np.float32)
    dst = np.array(target_pts, dtype=np.float32)
    
    # For backward mapping: we want to find, for each output pixel v, 
    # the corresponding input pixel u. So we treat dst as points in output space
    # and src as points in input space.
    # The mapping f(v) gives the source coordinate u.
    # Following MLS paper, we use the same formulas but swap src and dst.
    # Actually, the paper defines f(v) = l_v(v) where l_v minimizes weighted sum.
    # To get inverse mapping, we swap roles of p and q.
    p = dst   # points in output space (control points after deformation)
    q = src   # corresponding points in input space
    
    # Prepare output image
    warped = np.zeros_like(image)
    
    # For each pixel in output image
    for y in range(h):
        for x in range(w):
            v = np.array([x, y], dtype=np.float32)
            
            # Compute weights
            dist2 = np.sum((p - v) ** 2, axis=1)
            w_i = 1.0 / (dist2 + eps)  # alpha = 1.0
            sum_w = np.sum(w_i)
            if sum_w == 0:
                # No influence, use identity mapping
                u = v
            else:
                # Weighted centroids
                p_star = np.sum(w_i[:, None] * p, axis=0) / sum_w
                q_star = np.sum(w_i[:, None] * q, axis=0) / sum_w
                
                # Shifted points
                p_hat = p - p_star
                q_hat = q - q_star
                
                # Compute f_r(v) = sum (q_hat_i * A_i)
                # where A_i = w_i * [p_hat_i, -p_hat_i_perp] * [v-p_star, -(v-p_star)^perp]^T
                delta = v - p_star
                delta_perp = np.array([-delta[1], delta[0]])
                
                # Initialize f_r as 2-vector
                f_r = np.zeros(2, dtype=np.float32)
                
                for i in range(n_pts):
                    # p_hat_i is 2-vector
                    p_i = p_hat[i]
                    p_perp = np.array([-p_i[1], p_i[0]])
                    
                    # Build 2x2 matrix M_i = [p_i, -p_perp]
                    # Build 2x2 matrix N = [delta, -delta_perp]
                    # Then A_i = w_i * M_i * N^T (2x2)
                    # Actually we need only (q_hat_i * A_i) = (q_hat_i * M_i) * N^T
                    # Compute (q_hat_i * M_i) which is 1x2 vector
                    # q_hat_i is 1x2 row vector
                    q_i = q_hat[i]
                    # q_i * M_i = [q_i · p_i,  q_i · (-p_perp)] = [q_i·p_i, -q_i·p_perp]
                    a = np.dot(q_i, p_i)
                    b = -np.dot(q_i, p_perp)
                    # Then multiply by N^T: [a, b] * [delta, delta_perp]^T = a*delta + b*delta_perp
                    f_r += w_i[i] * (a * delta + b * delta_perp)
                
                # Normalize to get u
                norm = np.linalg.norm(f_r)
                if norm > eps:
                    u = q_star + np.linalg.norm(delta) * (f_r / norm)
                else:
                    u = q_star  # fallback
            
            # Sample source image at u (bilinear interpolation)
            ux, uy = u[0], u[1]
            # Clamp to image boundaries
            ux = np.clip(ux, 0, w - 1)
            uy = np.clip(uy, 0, h - 1)
            
            # Bilinear interpolation
            x0 = int(np.floor(ux))
            y0 = int(np.floor(uy))
            x1 = min(x0 + 1, w - 1)
            y1 = min(y0 + 1, h - 1)
            dx = ux - x0
            dy = uy - y0
            
            # Interpolate for each channel
            if len(image.shape) == 2:  # grayscale
                val = (1 - dx) * (1 - dy) * image[y0, x0] + \
                      dx * (1 - dy) * image[y0, x1] + \
                      (1 - dx) * dy * image[y1, x0] + \
                      dx * dy * image[y1, x1]
                warped[y, x] = val
            else:  # color
                for c in range(image.shape[2]):
                    val = (1 - dx) * (1 - dy) * image[y0, x0, c] + \
                          dx * (1 - dy) * image[y0, x1, c] + \
                          (1 - dx) * dy * image[y1, x0, c] + \
                          dx * dy * image[y1, x1, c]
                    warped[y, x, c] = val
    
    return warped.astype(image.dtype)

def run_warping():
    global points_src, points_dst, image
    if len(points_src) != len(points_dst) or len(points_src) < 3:
        # Not enough points, return original image
        return image
    
    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))
    return warped_image

# Clear all selected points
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image

# Build Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image", interactive=True, width=800)
            point_select = gr.Image(label="Click to Select Source and Target Points", interactive=True, width=800)

        with gr.Column():
            result_image = gr.Image(label="Warped Result", width=800)

    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")

    input_image.upload(upload_image, input_image, point_select)
    point_select.select(record_points, None, point_select)
    run_button.click(run_warping, None, result_image)
    clear_button.click(clear_points, None, point_select)

demo.launch()

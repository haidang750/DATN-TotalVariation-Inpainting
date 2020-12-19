import numpy as np
from numpy import clip, empty,  Inf, mod, sum, vstack, zeros
from numpy.linalg import norm
import cvxpy as cp

np.random.seed(21)

def L2_Norm_Total_Variation(image):
    rows, cols = image.shape

    col_diff = image[: -1, 1 :] - image[: -1, : -1] # pixel(i,j+1) - pixel(i,j)        shape: (255, 255)
    row_diff = image[1 :, : -1] - image[: -1, : -1] # pixel(i+1,j) - pixel(i,j)        shape: (255, 255)

    # vstack((col_diff.T.flatten(), row_diff.T.flatten())).T      # shape: (255*255, 2)

    # Tính giá trị L2-norm total variation của hình ảnh
    diff_norms = norm(vstack((col_diff.T.flatten(), row_diff.T.flatten())).T, ord=2, axis=1)   # shape: (255*255, 1)
    value = sum(diff_norms) / ((rows - 1) * (cols - 1)) # giá trị biến phân tổng (total variation)

    subgrad = zeros((rows, cols))
    norms_mat = diff_norms.reshape(cols - 1, rows - 1).T
    norms_mat[norms_mat == 0] = Inf # nếu có bất kỳ norm_mat nào trong ba mẫu số bằng 0 thì subgrad = 0 (phép chia cho vô cùng)
    subgrad[: -1, : -1] = - col_diff / norms_mat                            # (Y(i,j) - Y(i,j+1))/norms_mat
    subgrad[: -1, 1 :] = subgrad[: -1, 1 :] + col_diff / norms_mat          # (Y(i,j+1) - Y(i,j))/norms_mat
    subgrad[: -1, : -1] = subgrad[: -1, : -1] - row_diff / norms_mat        # (2Y(i,j) - Y(i+1,j) - Y(i,j+1))/norms_mat
    subgrad[1 :, : -1] = subgrad[1 :, : -1] + row_diff / norms_mat          # (Y(i+1,j) - Y(i,j))/norms_mat

    return value, subgrad

# alpha: tử số trong square-summable-but-not-summable (SSBNS) step size
# beta:  mẫu số trong SSBNS step size
def inpaintProcess(image, mask, alpha, beta):
    max_iter = 1000            # số bước lặp (iteration)
    painted = image
    painted_best = image
    obj_best = Inf

    # apply Subgradient Method
    for n_iter in range(max_iter):
        # obj là giá trị biến phân tổng của hình ảnh painted, subgrad là tập các đạo hàm thành phần tại tất cả các điểm ảnh của hình ảnh painted
        obj, subgrad = L2_Norm_Total_Variation(painted)

        if obj < obj_best:                                                  # f(k)(best) = min{ f(k-1)(best), f(x(k)) }
            obj_best = obj
            painted_best = painted

        painted_prev = painted
        painted = painted - (alpha/ (beta + n_iter)) * subgrad

        painted[mask] = image[mask]
        clip(painted, 0, 256, painted)

        # Check for convergence (độ hội tụ)
        if norm(painted - painted_prev) / norm(painted) < 1e-3:
            break
    painted = painted_best
    return painted, obj_best

def getRecoveredImage(originalImage, damagedImage, alpha, beta):
    originalImage = originalImage.astype(float)
    rows, cols, colors = originalImage.shape

    # mask
    mask = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            for k in range(colors):
                if (originalImage[i, j, k] == damagedImage[i, j, k]):
                    mask[i, j] = 1
    mask = mask.astype(bool)

    recoveredImage = empty(originalImage.shape)
    for i in range(colors):
        recoveredImage[:, :, i], obj_best = inpaintProcess(originalImage[:, :, i], mask, alpha, beta)
    print("Total variation value of Recovered Image = ", obj_best) # giá trị biến phân tổng của hình ảnh khôi phục
    return recoveredImage

# available tv_inpainting from CVXPY library
def availableInpainting(originalImage, damagedImage):
    rows, cols, colors = originalImage.shape
    mask = zeros((rows, cols, colors))

    for i in range(rows):
        for j in range(cols):
            for k in range(colors):
                if (originalImage[i, j, k] == damagedImage[i, j, k]):
                    mask[i, j, k] = 1

    variables = []
    constraints = []

    for i in range(colors):
        U = cp.Variable(shape=(rows, cols))
        variables.append(U)
        constraints.append(cp.multiply(mask[:, :, i], U) == cp.multiply(mask[:, :, i], damagedImage[:, :, i]))
    prob = cp.Problem(cp.Minimize(cp.tv(*variables)), constraints)
    prob.solve(verbose=True, solver=cp.SCS)

    recoveredImage = zeros((rows, cols, colors))
    for i in range(colors):
        recoveredImage[:, :, i] = variables[i].value

    recoveredImage = clip(recoveredImage, 0, 256, recoveredImage)
    return recoveredImage


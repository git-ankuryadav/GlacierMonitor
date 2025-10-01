import numpy as np
data1 = np.load("npyy/slice_0_img_034.npy")
# print(data1)
# print(data1.shape)
# print(data1.ndim)

# mask1 = np.load("npyy/slice_1_mask_056.npy")
mask1 = np.load("npyy/mask_27.npy")
print(mask1)
print(mask1.shape)
print(mask1.ndim)

channels_to_select = [1, 2, 3, 5, 9]
selected_data = data1[:, :, channels_to_select]
# print(selected_data.shape)
# print(selected_data)



# for ch in range(data1.shape[2]):
#     print(f"Channel {ch}: NAN count = {np.isnan(data1[:, :, ch]).sum()}")

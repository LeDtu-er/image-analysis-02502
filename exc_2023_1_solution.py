im_name = r"1-166.dcm"
Liver_ROI = io.imread(
    r"LiverROI.png"
).astype(bool)
Kidney_ROI_r = io.imread(
    r"KidneyRoi_r.png"
).astype(bool)
Kidney_ROI_l = io.imread(
    r"KidneyRoi_l.png"
).astype(bool)

# Use the dicom import to read the files
ds = dicom.read_file(im_name)
img_dcm = ds.pixel_array

io.imshow(img_dcm, cmap="gray")

# Vi tager det originale CT billede og finder de værdier som passer for hvert af de tre billeder
img_liver = img_dcm[Liver_ROI]
img_kidney_r = img_dcm[Kidney_ROI_r]
img_kidney_l = img_dcm[Kidney_ROI_l]

print(np.mean(img_kidney_l), np.mean(img_kidney_r), np.std(img_kidney_r),np.std(img_kidney_l))
t1 = np.mean(img_liver)- np.std(img_liver)
t2 = np.mean(img_liver)+ np.std(img_liver)


img_bin = (img_dcm > t1) & (img_dcm < t2)

img_bin = dilation(img_bin,disk(3))
img_bin = erosion(img_bin,disk(10))
img_bin = dilation(img_bin,disk(10))

#img_bin = segmentation.clear_border(img_bin)
label_img = measure.label(img_bin)
image_label_overlay = label2rgb(label_img)


# forskellige measures:
region_props = measure.regionprops(label_img)
areas = np.array([prop.area for prop in region_props])
perimeter = np.array([prop.perimeter for prop in region_props])
circularity = np.array([4 * math.pi * A / P**2 for (A, P) in zip(areas, perimeter)])


# for at fjerne ting der ikke lever op til nogle thresholds for area/perimeter/circ brug denne funktion med satte thresholds:
min_area = 1500
max_area = 7000
min_perim = 300
# .... osv

# Create a copy of the label_img
label_img_filter = label_img
for region in region_props:
    # (add other criteria down here!! ->)
    if region.area > max_area or region.area < min_area or region.perimeter < min_perim:
        # set the pixels in the invalid areas to background
        for cords in region.coords:
            label_img_filter[cords[0], cords[1]] = 0

# Create binary image from the filtered label image
i_area = label_img_filter > 0
show_comparison(img_bin, i_area)


ground_truth_img = Liver_ROI
gt_bin = ground_truth_img > 0

dice_score = 1 - distance.dice(i_area.ravel(), gt_bin.ravel())
print(f"DICE score {dice_score}")
print(t1,t2)

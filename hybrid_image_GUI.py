import argparse
import PySimpleGUI as sg
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import yaml

def np_im_to_data(im):
    array = np.array(im, dtype=np.uint8)
    im = Image.fromarray(array)
    with BytesIO() as output:
        im.save(output, format='PNG')
        data = output.getvalue()
    return data

def construct_image_histogram(np_image):
    L = 256
    bins = np.arange(L+1)
    hist, _ = np.histogram(np_image, bins)
    return hist

def draw_hist(canvas, figure):
   canvas.delete("all")

   tkcanvas = FigureCanvasTkAgg(figure, canvas)
   tkcanvas.draw()
   tkcanvas.get_tk_widget().pack(side='top', fill='both', expand=1)

def display_image(np_image_1, np_image_2):
    # Make sure image 2 is the same size as image 1 before using
    if np_image_1.shape[:2] != np_image_2.shape[:2]:
        np_image_2 = cv2.resize(np_image_2, (np_image_1.shape[1], np_image_1.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Convert numpy array to data that sg. Graph can understand
    image_data_1 = np_im_to_data(np_image_1)
    image_data_2 = np_im_to_data(np_image_2)

    original_image_1 = np_image_1.copy()  # Store a copy of the original image for reset
    original_image_2 = np_image_2.copy()

    filtered_image_1 = np_image_1.copy()  # Initialize filtered image
    filtered_image_2 = np_image_2.copy()
      
    final_filtered_image = np_image_1.copy()

    layout = [
        [sg.Push(), sg.Image(data=image_data_1, key='IMAGE_1'), sg.Image(data=image_data_2, key='IMAGE_2'), sg.Image(data=image_data_1, key='IMAGE_3'), sg.Push()],
       
        [sg.Text("Low Pass "),
         sg.Slider((1, 50), default_value=1, size=(113,20), orientation='horizontal', expand_x=True, key="LOW"),
         sg.Button('Save Settings'), sg.Button('Load Settings')],

        [sg.Text("High Pass"),
         sg.Slider((1, 50), default_value=1, size=(114,20), orientation='horizontal', expand_x=True, key="HIGH"),
         sg.Button('Save Image'), sg.Button('Load Image')],

        [sg.Text("Histogram Equalization"), sg.Checkbox('',key='H_EQUALIZED'), 
         sg.Text("Grayscale"), sg.Checkbox('', key='GRAYSCALE'),
         sg.Push(), sg.Button("Image Histograms"), sg.Button('Apply Filters'), sg.Button('Reset'), sg.Button('Quit')],
    ]

    # Create the window
    window = sg.Window('Display Image', layout, finalize=True)

    # Event loop
    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED or event == 'Quit':
            break

        # Get values from sliders
        low_value = int(values['LOW'])
        high_value = int(values['HIGH'])
        heq_value = values['H_EQUALIZED']
        is_grayscale = values['GRAYSCALE']
    
        # When load settings button is clicked
        if event == "Load Settings":
            values = open_load_settings()

            window['LOW'].update(value=int(values['LOW']))
            window['HIGH'].update(value=int(values['HIGH']))
            window['H_EQUALIZED'].update(value=values['H_EQUALIZED'])

        # When load image button is clicked
        if event == 'Load Image':
            # User changes first image
            new_image_1 = open_load_image()
            if new_image_1 is not None:
                original_image_1 = new_image_1.copy()
                filtered_image_1 = new_image_1.copy()
                final_filtered_image = new_image_1.copy()

                new_image_data = np_im_to_data(original_image_1)
                window['IMAGE_1'].update(data=new_image_data)
                window['IMAGE_3'].update(data=new_image_data)
                
            # User changes second image
            new_image_2 = open_load_image()
            if new_image_2 is not None:
                original_image_2 = new_image_2.copy()
                
                new_image_data_2 = np_im_to_data(original_image_2)
                window['IMAGE_2'].update(data=new_image_data_2)

        # When save settings button is clicked
        if event == "Save Settings":
            save_settings(low_value, high_value, heq_value)

        # When save image button is clicked
        if event == 'Save Image':
            save_image(final_filtered_image)

        if event == 'Apply Filters':
            # Apply Low Pass filter to image 1 if the slider value is greater than 0
            if low_value > 0:
                low_pass_filter = create_low_filter(low_value)
                filtered_image_1 = apply_filter_to_image(filtered_image_1, low_pass_filter)
            # Apply High Pass filter to image 2 if the slider value is greater than 0
            if high_value > 0:
                high_pass_filter = create_high_filter(high_value)
                filtered_image_2 = apply_filter_to_image(filtered_image_2, high_pass_filter)

            # Add images together to get final filtered image
            final_filtered_image = create_hybrid_image(filtered_image_1, filtered_image_2)

            # Apply histogram equalization to final image if checked
            if heq_value:
                final_filtered_image = hist_equalization(final_filtered_image)

            # Apply grayscale to final image if checked
            if is_grayscale:
                final_filtered_image = img_to_grayscale(final_filtered_image)
                
            # Update final image and display it
            ff_image_data = np_im_to_data(final_filtered_image)
            window['IMAGE_3'].update(data=ff_image_data)

        # Show Image Histograms
        if event == 'Image Histograms':
            create_histogram_popup(original_image_1, original_image_2, final_filtered_image)

        # Handle reset button
        if event == 'Reset':
            window['HIGH'].update(value=0)
            window['LOW'].update(value=0)
            window['H_EQUALIZED'].update(value=False)
            window['GRAYSCALE'].update(value=False)
            filtered_image_1 = original_image_1.copy()  # Reset to original image
            filtered_image_2 = original_image_2.copy()
            final_filtered_image = original_image_1.copy()
            reset_image_data = np_im_to_data(final_filtered_image)
            window['IMAGE_3'].update(data=reset_image_data)

    window.close()

# Functions to load and save settings
def open_load_settings():
    settings_file = sg.popup_get_file('Load Settings', file_types=(("YAML Files", "*.yaml"), ("All Files", "*.*")))
    
    if settings_file:
        with open(settings_file, 'r') as file:
            settings = yaml.safe_load(file)
            return settings
    return dict(LOW = 0, HIGH = 0, H_EQUALIZED = False)

def save_settings(low_value, high_value, heq_value):
    settings = dict(
        LOW = low_value,
        HIGH = high_value,
        H_EQUALIZED = heq_value
    )

    settings_file = sg.popup_get_file('Save Settings', save_as=True, file_types=(("YAML Files", "*.yaml"), ("All Files", "*.*")))

    if settings_file:
        with open('settings.yaml', 'w') as outfile:
            yaml.dump(settings, outfile, default_flow_style=False)
            sg.popup("Settings Saved Successfully!")

# Functions to open and save an image
def open_load_image():
    # Use a file dialog to select an image
    image_path = sg.popup_get_file('Open Image', file_types=(("Image Files", "*.png;*.jpg;*.jpeg"), ("All Files", "*.*")))

    if not image_path:
        sg.popup("No file selected. Exiting...")
        return None

    # Load and display the image
    image = cv2.imread(image_path)
    if image is None:
        sg.popup_error(f"Error: Unable to open {image_path}")
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = resize_img_with_ratio(image, 640, 480)

    return image

def save_image(image):
    # Popup to get the file save path
    file_path = sg.popup_get_file('Save As', save_as=True, no_window=True, file_types=(("PNG Files", "*.png"), ("JPEG Files", "*.jpg"), ("All Files", "*.*")))

    if file_path:
        try:
            # Save the image directly if already in BGR format, or convert otherwise
            if len(image.shape) == 3 and image.shape[2] == 3:
                edited_image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR for saving
            else:
                edited_image_bgr = image  # Assume grayscale if no color channels

            cv2.imwrite(file_path, edited_image_bgr)
            sg.popup("Image saved successfully!")
        except Exception as e:
            sg.popup_error(f"Error saving image: {e}")
    else:
        print("Save operation was cancelled")

# Maintain the aspect ratio
def resize_img_with_ratio(image,target_width,target_height):
    original_height, original_width = image.shape[:2]
    aspect_ratio = original_width / original_height
    
    if original_width > original_height:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(target_height * aspect_ratio)
    
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return resized_image


# Low Pass Filter - Using Gaussian Blur
def create_low_filter(half_width):
    filter_size = 2 * half_width + 1
    sigma = half_width / 3

    # Generate the 1D Gaussian kernel
    gaussian_1d = cv2.getGaussianKernel(filter_size, sigma)

    # Compute the outer product to get the 2D Gaussian kernel
    gaussian_2d = np.outer(gaussian_1d, gaussian_1d)

    return gaussian_2d

# High Pass Filter - Using Guassian Blur then subtracting from original image
def create_high_filter(half_width):
    filter_size = 2 * half_width + 1
    sigma = half_width / 3

    # Generate the 1D Gaussian kernel
    gaussian_1d = cv2.getGaussianKernel(filter_size, sigma)

    # Compute the outer product to get the 2D Gaussian kernel
    gaussian_2d = np.outer(gaussian_1d, gaussian_1d)

    # Create an identity filter (all zeros except the center is 1)
    identity_filter = np.zeros_like(gaussian_2d)
    center = filter_size // 2
    identity_filter[center, center] = 1

    # Subtract the Gaussian blur kernel from the identity filter to get the high-pass filter
    high_pass_filter = identity_filter - gaussian_2d

    return high_pass_filter

# Method that applies an h × w filter to an h × w patch. Returns a scalar value
def apply_filter_to_patch(patch, filter):
    # Multiply the filter and patch
    return np.sum(patch * filter)

# Method which returns an image (minus the boundary pixels). Internally this method uses your own apply_filter_to_patch method
def apply_filter_to_image(image, filter):
    # Handle RGBA separately, we will filter only RGB channels
    if image.shape[2] == 4:  # Image has an alpha channel
        rgb_image = image[..., :3]  # Extract RGB part
        alpha_channel = image[..., 3]  # Keep the alpha channel separate
    else:
        rgb_image = image
        alpha_channel = None

    img_h, img_w = rgb_image.shape[:2]
    f_h, f_w = filter.shape[:2]

    # Calculate half sizes of the filter (to deal with boundaries)
    half_h = f_h // 2
    half_w = f_w // 2

    # create output image without boundary pixels
    output_image = np.zeros((img_h - 2 * half_h, img_w - 2 * half_w, 3))

    # loop through each section of image
    for i in range(half_h, img_h - half_h):
        for j in range(half_w, img_w - half_w):
            for k in range(3):  # Apply the filter to each RGB channel
                patch = rgb_image[i - half_h: i + half_h + 1, j - half_w: j + half_w + 1, k]
                output_image[i - half_h, j - half_w, k] = apply_filter_to_patch(patch, filter)

    if alpha_channel is not None:
        # Resize alpha channel to match the filtered image and concatenate
        alpha_channel_resized = alpha_channel[half_h:img_h - half_h, half_w:img_w - half_w]
        output_image = np.dstack((output_image, alpha_channel_resized))

    return output_image

# Combine the images to create final hybrid image
def create_hybrid_image(low_pass_image, high_pass_image):
    if low_pass_image.shape != high_pass_image.shape:
        # Resize high_pass_image to match the size of low_pass_image
        high_pass_image = cv2.resize(high_pass_image, (low_pass_image.shape[1], low_pass_image.shape[0]))

    hybrid_image = low_pass_image + high_pass_image

    # return hybrid image, ensuring pixel values stay within 0 - 255 range
    return np.clip(hybrid_image, 0, 255).astype(np.uint8)


# perform histogram Equalization
def histogram_equalization_numpy(image):
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    equalized_image = cdf[image]
    return equalized_image

def hist_equalization(image):
    # Convert RGB to HSV
    hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Apply histogram equalization to V Channel only
    # First split the channels
    h,s,v = cv2.split(hsv_img)

    # Then apply histogram equalization to the V channel
    equalized_v = histogram_equalization_numpy(v)

    # Merge channels back together
    hsv_equalized = cv2.merge([h,s,equalized_v])

    # Convert back to RGB
    equalized_img = cv2.cvtColor(hsv_equalized, cv2.COLOR_HSV2RGB)

    return equalized_img


# changes image to grayscale
def img_to_grayscale(image):
    grayscale_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return grayscale_img

# Helper method to create window that displays image histograms
def create_histogram_popup(image_1, image_2, image_hybrid):
        # Construct histograms for each image
        hist_1 = construct_image_histogram(cv2.cvtColor(image_1, cv2.COLOR_RGB2GRAY))
        hist_2 = construct_image_histogram(cv2.cvtColor(image_2, cv2.COLOR_RGB2GRAY))
        hist_hybrid = construct_image_histogram(cv2.cvtColor(image_hybrid, cv2.COLOR_RGB2GRAY))
    
        # Create figures for each histogram
        fig_1 = plt.figure(figsize=(6, 6), dpi=100)
        ax1 = fig_1.add_subplot(111)
        ax1.bar(np.arange(len(hist_1)), hist_1, color='blue')
        ax1.set_title('Image 1 Histogram')

        fig_2 = plt.figure(figsize=(6, 6), dpi=100)
        ax2 = fig_2.add_subplot(111)
        ax2.bar(np.arange(len(hist_2)), hist_2, color='green')
        ax2.set_title('Image 2 Histogram')

        fig_hybrid = plt.figure(figsize=(6, 6), dpi=100)
        ax_hybrid = fig_hybrid.add_subplot(111)
        ax_hybrid.bar(np.arange(len(hist_hybrid)), hist_hybrid, color='red')
        ax_hybrid.set_title('Hybrid Image Histogram')
            
        # Create Popup window
        layout = [
            [sg.Canvas(key='HIST_1', size=(500, 400)),
            sg.Canvas(key='HIST_2', size=(500, 400)),
            sg.Canvas(key='HIST_HYBRID', size=(500, 400))],
            [sg.Push(),sg.Button('Close', size=(10, 2), font=('Arial', 16)),sg.Push()]
        ]

        histogram_window = sg.Window('Histogram Window', layout, finalize=True, size=(1850,700), modal=True)
        
        # Draw histograms on the canvases
        draw_hist(histogram_window['HIST_1'].TKCanvas, fig_1)
        draw_hist(histogram_window['HIST_2'].TKCanvas, fig_2)
        draw_hist(histogram_window['HIST_HYBRID'].TKCanvas, fig_hybrid)

        # Histogram Event loop
        while True:
            histogram_event, histogram_values = histogram_window.read()

            # if cancel is selected
            if histogram_event == sg.WIN_CLOSED or histogram_event == 'Close':
                histogram_window.close()
                break


def main():
    # Use a file dialog to select the first image
    image_path_1 = sg.popup_get_file('Open First Image', file_types=(("Image Files", "*.png;*.jpg;*.jpeg"), ("All Files", "*.*")))

    if not image_path_1:
        sg.popup("No file selected. Exiting...")
        return

    # Load and display the image
    image_1 = cv2.imread(image_path_1)
    if image_1 is None:
        sg.popup_error(f"Error: Unable to open {image_path_1}")
        return
    image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
    image_1 = resize_img_with_ratio(image_1, 640, 480)
    print(f'{image_1.shape}')
    

    # Use a file dialog to select the second image
    image_path_2 = sg.popup_get_file('Open Second Image', file_types=(("Image Files", "*.png;*.jpg;*.jpeg"), ("All Files", "*.*")))

    if not image_path_2:
        sg.popup("No file selected. Exiting...")
        return

    # Load and display the image
    image_2 = cv2.imread(image_path_2)
    if image_2 is None:
        sg.popup_error(f"Error: Unable to open {image_path_2}")
        return
    image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)
    image_2 = resize_img_with_ratio(image_2, 640, 480)
    print(f'{image_2.shape}')
    
    
    display_image(image_1, image_2)    

if __name__ == '__main__':
    main()

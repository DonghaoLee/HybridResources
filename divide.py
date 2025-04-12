from PIL import Image, ImageChops

def trim_whitespace(im, bg_color=(255,255,255), margin=10):
    """
    Trims away all background `bg_color` around the figure in `im`,
    then adds `margin` pixels of extra whitespace on each side.
    Returns the trimmed (and padded) image.
    """
    if im.mode != 'RGB':
        im = im.convert('RGB')
    bg = Image.new(im.mode, im.size, bg_color)
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if bbox is None:
        # entire image is background
        return im

    # Crop to bounding box
    cropped = im.crop(bbox)
    # Create new blank image to add margin
    new_w = cropped.width  + 2 * margin
    new_h = cropped.height + 2 * margin
    new_im = Image.new('RGB', (new_w, new_h), bg_color)
    new_im.paste(cropped, (margin, margin))
    return new_im


def manually_split_figure(
    in_path='Fig_all.png',
    out_prefix='Fig_part_',
    x_divisions=(0, 600, 1200, 1800, 2400),
    trim=True,
    pad_to_same_size=True,
    bg_color=(255,255,255),
    margin=10
):
    """
    1) Loads the figure from 'in_path'.
    2) Splits horizontally according to 'x_divisions', which must be an
       increasing sequence of x-coordinates, e.g. (0, 600, 1200, 1800, 2400).
       The i-th subplot will be cropped from (x_divisions[i], 0, x_divisions[i+1], height).
    3) Optionally trims whitespace around each subplot (if trim=True).
    4) Optionally pads them so they all have the same size (if pad_to_same_size=True).
    5) Saves them with filenames like 'Fig_part_0.png', 'Fig_part_1.png', etc.
    """

    # Load the combined figure
    img = Image.open(in_path)
    W, H = img.size
    print(W, H)

    # Sanity check: last x_division must not exceed total width
    if x_divisions[-1] > W:
        raise ValueError("x_divisions go beyond image width. Check your coordinates.")

    # Crop each region
    sub_imgs = []
    for i in range(len(x_divisions) - 1):
        left  = x_divisions[i]
        right = x_divisions[i+1]
        box   = (left, 0, right, H)
        sub_img = img.crop(box)

        # Trim whitespace if requested
        if trim:
            sub_img = trim_whitespace(sub_img, bg_color=bg_color, margin=margin)

        sub_imgs.append(sub_img)

    # Optionally pad all to same final size
    if pad_to_same_size:
        max_w = max(im.width  for im in sub_imgs)
        max_h = max(im.height for im in sub_imgs)
        standardized = []
        for im in sub_imgs:
            new_im = Image.new('RGB', (max_w, max_h), bg_color)
            offset_x = (max_w - im.width) // 2
            offset_y = (max_h - im.height) // 2
            new_im.paste(im, (offset_x, offset_y))
            standardized.append(new_im)
        sub_imgs = standardized

    # Save each subplot
    for i, s in enumerate(sub_imgs):
        s.save(f"{out_prefix}{i}.png")
    print("Finished splitting and saving subplots.")


if __name__ == "__main__":
    # EXAMPLE USAGE:
    # Suppose you manually determined these boundary x-coordinates:
    #   * Subplot 0: [0, 600)
    #   * Subplot 1: [600, 1200)
    #   * Subplot 2: [1200, 1800)
    #   * Subplot 3: [1800, 2400)
    # Adjust them to match your actual figure’s pixel widths.
    x_coords = (0, 4600, 9464, 4600 + 9464, 18928)

    manually_split_figure(
        in_path='Fig_all.png',      # your combined figure
        out_prefix='movie_',     # output file prefix
        x_divisions=x_coords,
        trim=True,                  # True => trim whitespace
        pad_to_same_size=True,      # True => unify final W/H
        bg_color=(255,255,255),
        margin=10
    )

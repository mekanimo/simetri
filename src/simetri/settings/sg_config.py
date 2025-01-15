"""User defined default values.
The sg_config.ini file would be created when you run sg.create_config('c:/your_dir_here/')
Then you can change the default values in the sg_config.ini file.

"""

import configparser
from os import path

import simetri.graphics as sg


defaults = sg.defaults
config_ini = f"""

; Configuration file for the Simetri Graphics library.
; This file is used to set default values for the library.
; To use factory default for a field you can use DEFAULT.
; Do not remove any lines from this file just change the values if needed.
; For general help see the documentation section "User Settings"
; or use sg.help('dist_tol') for specific items.
; To reset the values to the factory defaults, use sg.reset_config_ini()
; If you have not created a config.ini file yet, this reset will fail.
; Use sg.create_config('c:/your_dir_here/') to create a config.ini file.
; If you create multiple config.ini files, the last one created will be used.
; These values can be changed temporarily from a script by using sg.defaults['key'] = value
; All default values can be changed temporarily by using sg.defaults['key'] = value
; To get a list of all default values, use list(sg.defaults.keys())
; If you have created a sg_config.ini file you can change a field permanently
; by using sg.set_config('key', 'value'). Both key and value must be strings.

[General Defaults]
    debug_mode = {sg.defaults['debug_mode']}
    ; If this is True, <job_name>_debug.txt will be created in the job directory.

    log_level = {sg.defaults['log_level']}
    ; DEBUG, INFO, WARNING, ERROR, CRITICAL

    save_with_versions = {sg.defaults['save_with_versions']}
    ; If True and file exists, the save function will add a version number
    ; to the file name and the job directory name with the version will be
    ; a sub-directory in the job directory.

    overwrite_files = {sg.defaults['overwrite_files']}
    ; If True, the save function will overwrite the file if it already exists


[File Paths]
    temp_dir = {sg.defaults['temp_dir']}
    job_dir = {sg.defaults['job_dir']}
    ; If you set the job_dir, then you can run jobs just by specifying the file name.
    ; canvas.save('my_file.pdf') will save the file in the job_dir.
    ; If you do not set the job_dir, you will need to specify the full path.
    ; canvas.save doesn't use the current working directory.
    ; Like all others, this can be set from a script by using
    ; sg.defaults['job_dir'] = 'c:/my_job_dir/'.
    ; This will be used for that script only and will not alter the config.ini file.
    ; To alter the config.ini location use sg.set_config('job_dir', 'c:/my_job_dir/')


[Tolerance Defaults]
    angle_atol = {sg.defaults['angle_atol']}
    angle_rtol = {sg.defaults['angle_rtol']}
    area_atol = {sg.defaults['area_atol']}
    area_rtol = {sg.defaults['area_rtol']}
    atol = {sg.defaults['atol']}
    rtol = {sg.defaults['rtol']}
    dist_tol = {sg.defaults['dist_tol']}


; All length units are in points (1/72 inch)
; All angles are in radians

[Style Defaults]
    back_style = {sg.defaults['back_style']}
    ; EMPTY, COLOR, SHADING, PATTERN, GRIDLINES
    ; This takes effect when shape.fill == True

    pattern_type = {sg.defaults['pattern_type']}
    ; BRICKS, CHECKERBOARD, CROSSHATCH, DOTS, CROSSHATCHDOTS, HORIZONTAL, VERTICAL
    ; FIVEPOINTEDSTARS, SIXPOINTEDSTARS, GRID, NORTHEAST, NORTHWEST, STARS, HATCH

    shade_type = {sg.defaults['shade_type']}
    ; AXIS_LEFT_RIGHT, AXIS_TOP_BOTTOM, AXIS_LEFT_MIDDLE, AXIS_RIGHT_MIDDLE, AXIS_TOP_MIDDLE,
    ; AXIS_BOTTOM_MIDDLE, BALL, BILINEAR, COLORWHEEL, COLORWHEEL_BLACK, COLORWHEEL_WHITE,
    ; RADIAL_INNER, RADIAL_OUTER, RADIAL_INNER_OUTER,

    marker_type = {sg.defaults['marker_type']}
    ; ASTERISK, BAR, CIRCLE, CROSS, DIAMOND, DIAMOND_F, EMPTY, FCIRCLE, HALFCIRCLE
    ; HALFCIRCLE_F, HALFDIAMOND, HALFDIAMOND_F, HALFSQUARE, HALFSQUARE_F, HEXAGON, HEXAGON_F
    ; INDICES, MINUS, OPLUS, OPLUS_F, OTIMES, OTIMES_F, PENTAGON, PENTAGON_F, PLUS SQUARE
    ; SQUARE_F STAR, STAR2, STAR3, TEXT, TRIANGLE, TRIANGLE_F

    fill_color = {sg.defaults['fill_color']}
    line_color = {sg.defaults['line_color']}
    ; You can use any named colors from the XKCD color survey
    ; https://xkcd.com/color/rgb/ or sg.help('colors')
    ; or use the RGB values like this: [255, 0, 0]
    ; you can also use sg.random_color  fill_color = sg.random_color
    ; Color palettes.
    ; For example fill_color = sg.palette('seq_ALGAE_256', step=8)
    ; Color palettes are adapted from palettable
    ; https://github.com/jiffyclub/palettable
    ; See the documentation for more information or use sg.help('palettes').

    shade_axis_angle = {sg.defaults['shade_axis_angle']}
    line_width = {sg.defaults['line_width']}
    line_dash_array = {sg.defaults['line_dash_array']}
    line_join = {sg.defaults['line_join']}
    ; BEVEL, MITER, ROUND

    line_cap = {sg.defaults['line_cap']}
    ; BUTT, ROUND, SQUARE

    line_miter_limit = {sg.defaults['line_miter_limit']}


[Geometry Defaults]
    circle_radius = {sg.defaults['circle_radius']}
    ellipse_width_height = {sg.defaults['ellipse_width_height']}
    rectangle_width_height = {sg.defaults['rectangle_width_height']}
    lace_offset = {sg.defaults['lace_offset']}

[Page Defaults]
    page_grid_back_color = {sg.defaults['page_grid_back_color']}
    page_grid_line_color = {sg.defaults['page_grid_line_color']}
    page_grid_line_width = {sg.defaults['page_grid_line_width']}
    page_grid_line_dash_array = {sg.defaults['page_grid_line_dash_array']}
    page_grid_spacing = {sg.defaults['page_grid_spacing']}
    page_grid_x_shift = {sg.defaults['page_grid_x_shift']}
    page_grid_y_shift = {sg.defaults['page_grid_y_shift']}

[tikz Defaults]
    keep_aux_files = {sg.defaults['keep_aux_files']}
    keep_tex_files = {sg.defaults['keep_tex_files']}
    ; if the Canvas.render == Render.TEX, the TeX file is kept by default

    keep_log_files = {sg.defaults['keep_log_files']}
    ; If the job did not compile, log files are kept by default

    tikz_nround = {sg.defaults['tikz_nround']}
    tikz_scale = {sg.defaults['tikz_scale']}

    latex_compiler = {sg.defaults['latex_compiler']}
    ; pdflatex, xelatex, lualatex
    ; If you are using font names, you should use xelatex or lualatex

    xelatex_run_options = {sg.defaults['xelatex_run_options']}
    ; Example: [-halt-on-error, -shell-escape, ...]
    ; You should not need to change this value unless you have a specific need.

    pdflatex_run_options = DEFAULT
    lualatex_run_options = DEFAULT
    ; DEFAULT means use the factory default value

[tikz Document Defaults]
    document_class = {sg.defaults['document_class']}
    ; article, standalone, beamer, book, report, etc.

    document_options = {sg.defaults['document_options']}
    use_packages = {sg.defaults['use_packages']}
"""


def reset_config_ini():
    create_config("c:/tmp/")


def create_config(dir_path: str):
    # Write the configuration to a file
    # combine dir_path and config.ini
    file_path = path.join(dir_path, "config.ini")
    with open(file_path, "w", encoding="utf-8") as config_file:
        config_file.write(config_ini)


def read_config(file_path: str):
    # Create a ConfigParser object
    config = configparser.ConfigParser()

    # Read the configuration file
    config.read(file_path)

    # Access values from the configuration file
    debug_mode = config.getboolean("General Defaults", "debug_mode")
    log_level = config.get("General Defaults", "log_level")
    pattern_type = config.get("Style Defaults", "pattern_type")
    use_packages = config.get("tikz Document Defaults", "use_packages")

    # config.getboolean, config.getint, config.getfloat

    # Return a dictionary with the retrieved values
    config_values = {
        "debug_mode": debug_mode,
        "log_level": log_level,
        "pattern_type": pattern_type,
        "use_packages": use_packages,
    }

    return config_values


create_config("c:/tmp/")

# Call the function to read the configuration file
config_data = read_config("c:/tmp/config.ini")

# Print the retrieved values
print("Debug Mode:", config_data["debug_mode"])
print("Log Level:", config_data["log_level"])
print("pattern_type:", config_data["pattern_type"])
print("use_packages:", config_data["use_packages"])

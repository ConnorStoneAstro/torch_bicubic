# torch_bicubic
Bicubic interpolation for PyTorch. This is a very simple bicubic interpolation
implementation taken from Numerical Recipes in C, Chapter 3 (specifically
"Higher Order for Smoothness: Bicubic Interpolaiton"). It is not expecially
efficient, though it should be reasonably performant. Note that this is not a
"bicubic spline" which is a specialized variant of bicubic interpolation. In the
function I allow the user to specify the necessary derivatives at each point,
though in the event that they are not supplied, the code will do a really basic
finite differences.

Note: If you want to use this in your research, pleae contact me. I'll have a
way to cite it figured out soon.

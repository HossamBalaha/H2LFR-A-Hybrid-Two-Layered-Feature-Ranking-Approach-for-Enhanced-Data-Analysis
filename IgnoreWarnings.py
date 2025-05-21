import warnings
import shutup

# Suppress all warnings globally.
warnings.filterwarnings("ignore")


def warn(*args, **kwargs):
  """
  Custom function to suppress warnings.
  This function overrides the default warning behavior to suppress all warnings.
  """
  pass


# Override the default warning function with the custom function.
warnings.warn = warn
shutup.please()  # Suppress all warnings using the `shutup` library.

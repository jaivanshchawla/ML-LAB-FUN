import os
import fastf1

# Make sure cache folder exists
os.makedirs("f1_cache", exist_ok=True)

# Enable cache
fastf1.Cache.enable_cache("f1_cache")

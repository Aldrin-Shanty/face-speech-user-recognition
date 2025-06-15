from audio_reg import audio_reg

# Run the function to record audio and extract features
features = audio_reg()

# Display the output
print("✅ Feature vector shape:", features.shape)
print("🎯 Feature vector:\n", features)

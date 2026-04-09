import tensorflow as tf
from tensorflow.keras.models import Model

print("🧠 Loading Original Music Model...")
# We use compile=False because we don't need its training settings anymore, just the brain structure.
original_model = tf.keras.models.load_model('Trained_model.h5', compile=False)

print("\n--- ORIGINAL ARCHITECTURE ---")
original_model.summary()

print("\n🔪 Performing Brain Surgery (Slicing off the final layer)...")
# We create a brand new model. 
# The input is the exact same.
# The output is hijacked to stop at the second-to-last layer (layers[-2]).
# Notice the 's' on inputs!
embedding_model = Model(inputs=original_model.inputs, 
                        outputs=original_model.layers[-2].output)

print("\n--- NEW EMBEDDING ARCHITECTURE ---")
embedding_model.summary()

# Save our new hacked model!
embedding_model.save('music_embedder.h5')
print("\n💾 Surgery Complete! New model saved as 'music_embedder.h5'")
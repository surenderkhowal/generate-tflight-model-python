package com.tensorflowtest

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class PlasticClassifier(private val context: Context) {
    private lateinit var interpreter: Interpreter
    private val labels = arrayOf(
        "LDPE Plastic Bag",
        "PET Plastic Bottle",
        "PS Styrofoam Box",
        "PP Plastic Cup",
        "PUR Surgical Gloves",
        "HDPE Plastic Jug",
        "PP Plastic Utensils",
    )

    init {
        // Load the TFLite model from assets
        val model = loadModelFile(context)
        interpreter = Interpreter(model)
    }

    private fun loadModelFile(context: Context): ByteBuffer {
        val fileDescriptor = context.assets.openFd("fine_tuned_model.tflite")
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    fun classifyImage(bitmap: Bitmap): String {
        // Preprocess the image to match the input shape of the model
        val inputSize = 180
        val inputBuffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3).apply {
            order(ByteOrder.nativeOrder())
            rewind()
        }

        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        for (y in 0 until inputSize) {
            for (x in 0 until inputSize) {
                val pixel = resizedBitmap.getPixel(x, y)
                inputBuffer.putFloat(((pixel shr 16) and 0xFF) / 255.0f)
                inputBuffer.putFloat(((pixel shr 8) and 0xFF) / 255.0f)
                inputBuffer.putFloat((pixel and 0xFF) / 255.0f)
            }
        }

        val outputBuffer = Array(1) { FloatArray(labels.size) }
        interpreter.run(inputBuffer, outputBuffer)

        val output = outputBuffer[0]
        val maxIndex = output.indices.maxByOrNull { output[it] } ?: -1
        return if (maxIndex != -1) labels[maxIndex] else "Unknown"
    }

}

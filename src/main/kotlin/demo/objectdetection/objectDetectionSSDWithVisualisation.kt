/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package demo.objectdetection

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.size
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color.Companion.Red
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.res.loadImageBitmap
import androidx.compose.ui.res.useResource
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.*
import androidx.compose.ui.window.Window
import getFileFromResource
import org.jetbrains.kotlinx.dl.api.inference.loaders.ONNXModelHub
import org.jetbrains.kotlinx.dl.api.inference.objectdetection.DetectedObject
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.dataset.image.ColorOrder
import org.jetbrains.kotlinx.dl.dataset.preprocessor.*
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.*
import org.jetbrains.skia.*
import java.io.File
import kotlin.math.abs

fun main() {
    val modelHub =
        ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
    val model = ONNXModels.ObjectDetection.SSD.pretrainedModel(modelHub)

    model.use { detectionModel ->
        println(detectionModel)

        val fileName = "detection/image2.jpg"
        val imageFile = getFileFromResource(fileName)
        val detectedObjects =
            detectionModel.detectObjects(imageFile = imageFile, topK = 20)

        detectedObjects.forEach {
            println("Found ${it.classLabel} with probability ${it.probability}")
        }

        visualise(imageFile, fileName, detectedObjects)
    }
}

private fun visualise(
    imageFile: File,
    fileName: String,
    detectedObjects: List<DetectedObject>
) {
    val preprocessing: Preprocessing = preprocess {
        load {
            pathToData = imageFile
            imageShape = ImageShape(null, null, 3)
            colorMode = ColorOrder.BGR
        }
        transformImage {
            resize {
                outputWidth = 1200
                outputHeight = 1200
            }
        }
        transformTensor {
            rescale {
                scalingCoefficient = 255f
            }
        }
    }

    val rawImage = preprocessing().first

    drawDetectedObjects(fileName, rawImage, ImageShape(1200, 1200, 3), detectedObjects)
}

private fun drawDetectedObjects(fileName: String, dst: FloatArray, imageShape: ImageShape, detectedObjects: List<DetectedObject>) {

    application {

        val image = remember {
            useResource(fileName, ::loadImageBitmap)
        }

        val width = imageShape.width!!.toInt()
        val height = imageShape.height!!.toInt()
        val windowState = rememberWindowState(width = width.dp, height = height.dp)

        val typeface = Typeface.makeFromName("Courier New", FontStyle.NORMAL)
        val font = Font(typeface, 30f)

        val paint = Paint().apply {
            color = Color.WHITE
        }

        Window(
            onCloseRequest = ::exitApplication,
            state = windowState,
            title = "Object detection"
        ) {

            Canvas(modifier = Modifier.size(width.dp, height.dp)) {

                drawImage(image, dstSize = IntSize(width, height))

                detectedObjects.forEach { detectedObject ->
                    val top = detectedObject.yMin * height
                    val left = detectedObject.xMin * width
                    val bottom = detectedObject.yMax * height
                    val right = detectedObject.xMax * width
                    if (abs(top - bottom) > 300 || abs(right - left) > 300) return@forEach

                    val yRect = bottom
                    val xRect = left

                    drawIntoCanvas {
                        it.nativeCanvas.drawString("${detectedObject.classLabel} : ${detectedObject.probability}",
                            xRect, yRect, font, paint)
                    }

                    drawRect(
                        color = Red,
                        topLeft = Offset(xRect, yRect),
                        size = Size(right - left, top - bottom),
                        style = androidx.compose.ui.graphics.drawscope.Stroke(3f),
                    )
                }
            }
        }
    }
}






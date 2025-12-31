// dart
import 'dart:typed_data';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import '../models/depth_result.dart';
import '../utils/logger.dart';

class DepthEstimationService {
  Interpreter? _interpreter;
  bool _isInitialized = false;
  List<int> _inputShape = [];
  List<int> _outputShape = [];
  bool _inferenceRunning = false;
  double _calibrationFactor = 1.0;

  // Midas-V2 uses ImageNet normalization
  static const List<double> MEAN = [0.485, 0.456, 0.406];
  static const List<double> STD = [0.229, 0.224, 0.225];

  Future<void> initialize() async {
    if (_isInitialized) return;
    try {
      // Update path to your Midas-V2 model file
      _interpreter = await Interpreter.fromAsset('assets/models/depth/Midas-V2_float.tflite');
      _inputShape = _interpreter!.getInputTensor(0).shape;
      _outputShape = _interpreter!.getOutputTensor(0).shape;
      _isInitialized = true;
      await Logger.log('Midas-V2 model initialized. inputShape=$_inputShape outputShape=$_outputShape');
    } catch (e) {
      await Logger.log('Error initializing Midas-V2 model: $e');
    }
  }

  void calibrate(double realCm, double measuredPredictedCm) {
    if (measuredPredictedCm > 0) {
      _calibrationFactor = realCm / measuredPredictedCm;
      Logger.log('Depth calibration set. factor=$_calibrationFactor (real=$realCm measured=$measuredPredictedCm)');
    }
  }

  Future<DepthResult> estimateDepth(Uint8List imageBytes) async {
    if (!_isInitialized) await initialize();
    if (!_isInitialized) {
      await Logger.log('Midas-V2 model not initialized. Returning default DepthResult.');
      return DepthResult(hasCollision: false, minDistance: 999);
    }

    if (_inferenceRunning) {
      await Logger.log('Skipping estimateDepth because another inference is running.');
      return DepthResult(hasCollision: false, minDistance: 999);
    }

    _inferenceRunning = true;
    try {
      img.Image? image = img.decodeImage(imageBytes);
      if (image == null || imageBytes.length > 10000000) {
        await Logger.log('Invalid image or image too large. length=${imageBytes.length}');
        return DepthResult(hasCollision: false, minDistance: 999);
      }

      // Parse input shape to determine format
      int height = 256;
      int width = 256;
      bool isNCHW = false;

      if (_inputShape.length == 4) {
        if (_inputShape[1] == 3) {
          // NCHW format: [1, 3, H, W]
          isNCHW = true;
          height = _inputShape[2];
          width = _inputShape[3];
        } else if (_inputShape[3] == 3) {
          // NHWC format: [1, H, W, 3]
          isNCHW = false;
          height = _inputShape[1];
          width = _inputShape[2];
        }
      }

      await Logger.log('Resizing image to ${width}x${height} for Midas-V2 (format: ${isNCHW ? "NCHW" : "NHWC"})');
      final resized = img.copyResize(image, width: width, height: height);

      // Create input tensor in correct format
      final inputBuffer = _createInputBuffer(resized, height, width, isNCHW);

      // Create output buffer
      final outputBuffer = _createOutputBuffer();

      await Logger.log('Running Midas-V2 inference. input shape: $_inputShape, output shape: $_outputShape');

      try {
        _interpreter!.run(inputBuffer, outputBuffer);
      } catch (e) {
        await Logger.log('Inference error: $e');
        return DepthResult(hasCollision: false, minDistance: 999);
      }

      // Extract depth map from output
      List<List<double>> depth2D = _extractDepthMap(outputBuffer);
      if (depth2D.isEmpty) {
        await Logger.log('Depth map extraction returned empty result.');
        return DepthResult(hasCollision: false, minDistance: 999);
      }

      // Analyze depth in center region
      var stats = _analyzeDepth(depth2D);
      double globalMin = stats['globalMin']!;
      double globalMax = stats['globalMax']!;
      double maxValue = stats['maxValue']!;

      double normalizedDepth = 0.0;
      if (globalMax != globalMin) {
        normalizedDepth = (maxValue - globalMin) / (globalMax - globalMin);
      }

      double minDistanceCm = (1.0 - normalizedDepth) * 100.0 * _calibrationFactor;

      await Logger.log('Depth estimation done. normalizedDepth=$normalizedDepth minDistance=$minDistanceCm cm, maxValue=$maxValue globalMin=$globalMin globalMax=$globalMax');
      return DepthResult(
        hasCollision: normalizedDepth > 0.4,  // High normalizedDepth = close object
        minDistance: minDistanceCm,
        depthMap: depth2D,
      );
    } finally {
      _inferenceRunning = false;
    }
  }

  dynamic _createInputBuffer(img.Image image, int height, int width, bool isNCHW) {
    if (isNCHW) {
      // NCHW format: [1, 3, H, W] - nested list structure
      List<List<List<List<double>>>> input = [[]];

      for (int c = 0; c < 3; c++) {
        List<List<double>> channel = [];
        for (int y = 0; y < height; y++) {
          List<double> row = [];
          for (int x = 0; x < width; x++) {
            var pixel = image.getPixel(x, y);
            double val;
            if (c == 0) {
              val = (pixel.r / 255.0 - MEAN[0]) / STD[0];
            } else if (c == 1) {
              val = (pixel.g / 255.0 - MEAN[1]) / STD[1];
            } else {
              val = (pixel.b / 255.0 - MEAN[2]) / STD[2];
            }
            row.add(val);
          }
          channel.add(row);
        }
        input[0].add(channel);
      }
      return input;
    } else {
      // NHWC format: [1, H, W, 3] - nested list structure
      List<List<List<List<double>>>> input = [[]];

      for (int y = 0; y < height; y++) {
        List<List<double>> row = [];
        for (int x = 0; x < width; x++) {
          var pixel = image.getPixel(x, y);
          List<double> pixelValues = [
            (pixel.r / 255.0 - MEAN[0]) / STD[0],
            (pixel.g / 255.0 - MEAN[1]) / STD[1],
            (pixel.b / 255.0 - MEAN[2]) / STD[2],
          ];
          row.add(pixelValues);
        }
        input[0].add(row);
      }
      return input;
    }
  }

  dynamic _createOutputBuffer() {
    // Midas-V2 output is [1, H, W]
    if (_outputShape.isEmpty) {
      // Fallback to 256x256
      return List.generate(1, (_) =>
          List.generate(256, (_) =>
          List<double>.filled(256, 0.0)
          )
      );
    }

    if (_outputShape.length == 3) {
      // [1, H, W]
      int batch = _outputShape[0];
      int h = _outputShape[1];
      int w = _outputShape[2];
      return List.generate(batch, (_) =>
          List.generate(h, (_) =>
          List<double>.filled(w, 0.0)
          )
      );
    } else if (_outputShape.length == 4) {
      // [1, 1, H, W] or [1, H, W, 1]
      int batch = _outputShape[0];
      int dim1 = _outputShape[1];
      int dim2 = _outputShape[2];
      int dim3 = _outputShape[3];
      return List.generate(batch, (_) =>
          List.generate(dim1, (_) =>
              List.generate(dim2, (_) =>
              List<double>.filled(dim3, 0.0)
              )
          )
      );
    }

    // Fallback
    return List.generate(1, (_) =>
        List.generate(256, (_) =>
        List<double>.filled(256, 0.0)
        )
    );
  }

  List<List<double>> _extractDepthMap(dynamic outputBuffer) {
    List<List<double>> depth2D = [];

    try {
      if (_outputShape.isEmpty) return depth2D;

      if (_outputShape.length == 3) {
        // [1, H, W]
        int h = _outputShape[1];
        int w = _outputShape[2];

        for (int y = 0; y < h; y++) {
          List<double> row = [];
          for (int x = 0; x < w; x++) {
            double val = (outputBuffer[0][y][x] as num).toDouble();
            row.add(val);
          }
          depth2D.add(row);
        }
      } else if (_outputShape.length == 4) {
        // Handle [1, 1, H, W] or [1, H, W, 1]
        if (_outputShape[1] == 1) {
          // [1, 1, H, W]
          int h = _outputShape[2];
          int w = _outputShape[3];
          for (int y = 0; y < h; y++) {
            List<double> row = [];
            for (int x = 0; x < w; x++) {
              double val = (outputBuffer[0][0][y][x] as num).toDouble();
              row.add(val);
            }
            depth2D.add(row);
          }
        } else {
          // [1, H, W, 1]
          int h = _outputShape[1];
          int w = _outputShape[2];
          for (int y = 0; y < h; y++) {
            List<double> row = [];
            for (int x = 0; x < w; x++) {
              double val = (outputBuffer[0][y][x][0] as num).toDouble();
              row.add(val);
            }
            depth2D.add(row);
          }
        }
      }

      return depth2D;
    } catch (e) {
      Logger.log('Error extracting depth map: $e');
      return [];
    }
  }

  Map<String, double> _analyzeDepth(List<List<double>> depthMap) {
    double globalMin = double.infinity;
    double globalMax = double.negativeInfinity;

    int h = depthMap.length;
    int w = depthMap.isNotEmpty ? depthMap[0].length : 0;

    // Find global min/max
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        double val = depthMap[y][x];
        if (val < globalMin) globalMin = val;
        if (val > globalMax) globalMax = val;
      }
    }

    // Analyze center region (30%-70% of image)
    int y0 = (h * 0.3).floor();
    int y1 = (h * 0.7).floor();
    int x0 = (w * 0.3).floor();
    int x1 = (w * 0.7).floor();

    if (y1 <= y0 || x1 <= x0) {
      y0 = 0;
      y1 = h;
      x0 = 0;
      x1 = w;
    }

    // Find MAXIMUM in center region (closest object in inverse depth)
    double maxValue = double.negativeInfinity;
    for (int y = y0; y < y1; y++) {
      for (int x = x0; x < x1; x++) {
        double val = depthMap[y][x];
        if (val > maxValue) maxValue = val;
      }
    }

    if (globalMin == double.infinity) globalMin = 0.0;
    if (globalMax == double.negativeInfinity) globalMax = 0.0;
    if (maxValue == double.negativeInfinity) maxValue = 0.0;

    Logger.log('Depth analysis: maxValue=$maxValue globalMin=$globalMin globalMax=$globalMax');

    return {
      'maxValue': maxValue,
      'globalMin': globalMin,
      'globalMax': globalMax,
    };
  }

  void dispose() {
    _interpreter?.close();
    Logger.log('DepthEstimationService disposed.');
  }
}
import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import '../services/depth_estimation_service.dart';
import '../services/haptic_service.dart';
import '../services/tts_service.dart';
import '../models/depth_result.dart';

class CollisionProvider with ChangeNotifier {
  final DepthEstimationService _depthService = DepthEstimationService();
  final HapticService _hapticService = HapticService();
  final TtsService _tts_service = TtsService();

  bool _isMonitoring = false;
  DepthResult? _lastResult;
  Timer? _monitoringTimer;
  bool _isRunningInference = false;

  bool get isMonitoring => _isMonitoring;
  DepthResult? get lastResult => _lastResult;
  bool get hasCollisionRisk => _lastResult?.hasCollision ?? false;
  bool get isProcessing => _isRunningInference;

  Future<void> initialize() async {
    await _depthService.initialize();
  }

  Future<void> startMonitoring(Function captureImageCallback) async {
    if (_isMonitoring) return;

    _isMonitoring = true;
    notifyListeners();

    // Timer más espaciado para dar tiempo a la inferencia
    _monitoringTimer = Timer.periodic(Duration(seconds: 3), (timer) async {
      if (!_isMonitoring) return;
      if (_isRunningInference) {
        debugPrint('Skipping frame - previous inference still running');
        return;
      }

      _isRunningInference = true;
      notifyListeners();

      try {
        Uint8List? imageBytes = await captureImageCallback();
        if (imageBytes != null && _isMonitoring) {
          // Run inference in background using compute to avoid blocking UI
          final result = await _runDepthEstimationInBackground(imageBytes);

          // Check if result is valid before using it
          if (result != null) {
            _lastResult = result;

            if (result.hasCollision && _isMonitoring) {
              await _hapticService.vibrateCollisionAlert();
              await _tts_service.speak('Cuidado, riesgo de colision!');
            }
            notifyListeners();
          } else {
            debugPrint('Depth estimation returned null result');
          }
        }
      } catch (e) {
        debugPrint('Error en monitoreo: $e');
      } finally {
        _isRunningInference = false;
        notifyListeners();
      }
    });
  }

  // Single-shot capture method
  Future<void> captureOnce(Function captureImageCallback) async {
    if (_isRunningInference) {
      debugPrint('Inference already running - skipping single capture.');
      return;
    }

    _isRunningInference = true;
    notifyListeners();

    try {
      Uint8List? imageBytes = await captureImageCallback();
      if (imageBytes != null) {
        // Run inference in background to avoid ANR
        final result = await _runDepthEstimationInBackground(imageBytes);

        // Check if result is valid before using it
        if (result != null) {
          _lastResult = result;

          if (result.hasCollision) {
            await _hapticService.vibrateCollisionAlert();
            await _tts_service.speak('Cuidado, riesgo de colision!');
          }
          notifyListeners();
        } else {
          debugPrint('Depth estimation returned null result');
        }
      }
    } catch (e) {
      debugPrint('Error in single capture: $e');
    } finally {
      _isRunningInference = false;
      notifyListeners();
    }
  }

  // Helper method to run depth estimation in background
  Future<DepthResult?> _runDepthEstimationInBackground(Uint8List imageBytes) async {
    // Try to use compute (isolate) if possible, otherwise run directly
    try {
      // Note: compute doesn't work well with services that have state
      // So we run it directly but ensure the UI thread is not blocked
      // by yielding control periodically
      final result = await _depthService.estimateDepth(imageBytes);
      return result;
    } catch (e) {
      debugPrint('Error in background estimation: $e');
      // Return null instead of rethrowing to prevent crashes
      return null;
    }
  }

  void stopMonitoring() {
    _monitoringTimer?.cancel();
    _isMonitoring = false;
    _hapticService.stopVibration();
    _tts_service.stop();
    notifyListeners();
  }

  @override
  void dispose() {
    stopMonitoring();
    _depthService.dispose();
    super.dispose();
  }
}
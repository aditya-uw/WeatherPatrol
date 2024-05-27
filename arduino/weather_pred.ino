#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BME680.h>
// #include <RTCZero.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/kernels/micro_ops.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"

#include "model.h"

// BME680 sensor setup
Adafruit_BME680 bme;

// Mean and scale values obtained from the StandardScaler
const float mean[] = {51.88701241, 81.17352496, 1009.03348166}; // Replace with actual mean values
const float scale[] = {8.78442094, 14.9878748 ,  7.32198109};

// global variables used for TensorFlow Lite (Micro)
tflite::MicroErrorReporter tflErrorReporter;

// pull in all the TFLM ops, you can remove this line and
// only pull in the TFLM ops you need, if would like to reduce
// the compiled size of the sketch.
tflite::AllOpsResolver tflOpsResolver;

const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

// Create a static memory buffer for TFLM, the size may need to
// be adjusted based on the model you are using
constexpr int tensorArenaSize = 8 * 1024;
byte tensorArena[tensorArenaSize];

void setup() {
  Serial.begin(9600);
  while (!Serial) {}

  // Initialize BME680
  if (!bme.begin()) {
    Serial.println("Could not find a valid BME680 sensor, check wiring!");
    while (1);
  }

  // Set up real-time clock
  // rtc.begin();
  // You can set the RTC time here if needed, for example:
  // rtc.setTime(0); // set time to epoch
  // rtc.setDate(5, 27, 2024); // set date to 1st January 2024

  // Load TensorFlow Lite model
  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model version mismatch!");
    while (1);
  }

  // Set up the interpreter
  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize, &tflErrorReporter);

  // Allocate memory for the model's input and output tensors
  if (tflInterpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    while (1);
  }

  // Get pointers for the model's input and output tensors
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);
}

// Function to scale the input data
void scaleInputData(float* inputData, int dataSize) {
  for (int i = 0; i < dataSize; i++) {
    inputData[i] = (inputData[i] - mean[i]) / scale[i];
  }
}

void loop() {
  // Read sensor data
  if (!bme.performReading()) {
    Serial.println("Failed to perform reading from BME680 sensor!");
    return;
  }

  float temperature = bme.temperature * (9.0/5.0) + 32;
  float pressure = bme.pressure / 100.0; // hPa
  float humidity = bme.humidity;
  
  Serial.print("Temperature = ");
  Serial.print(temperature);
  Serial.println(" *F");

  Serial.print("Pressure = ");
  Serial.print(pressure);
  Serial.println(" hPa");

  Serial.print("Humidity = ");
  Serial.print(humidity);
  Serial.println(" %");

  // Prepare the input data array
  float inputData[] = {temperature, humidity, pressure};

  // Scale the input data
  scaleInputData(inputData, 3);

  // Prepare the input tensor
  tflInputTensor->data.f[0] = temperature;
  tflInputTensor->data.f[1] = humidity;
  tflInputTensor->data.f[2] = pressure;

  // Run inference
  if (tflInterpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed");
    return;
  }

  // Get the prediction results from the output tensor
  float no_rain = tflOutputTensor->data.f[0];
  float light_rain = tflOutputTensor->data.f[1];
  float heavy_rain = tflOutputTensor->data.f[2];

  // Print the results
  Serial.print("No Rain: "); Serial.println(no_rain);
  Serial.print("Light Rain: "); Serial.println(light_rain);
  Serial.print("Heavy Rain: "); Serial.println(heavy_rain);

  // Wait for 60 minutes (30 * 60 * 1000 milliseconds)
  delay(10000); // 60 minutes
}
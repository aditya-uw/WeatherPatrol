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
#include <ArduinoBLE.h>

#include "2class_model_for30.h"

// BME680 sensor setup
Adafruit_BME680 bme;

// BLE setup
BLEService weatherService("180C");
BLECharacteristic weatherCharacteristic("2A57", BLERead | BLENotify, 32); // Increase to 32 to handle longer strings

// Hard-coded month value (e.g., May)
const int month = 5; // Change to the desired month

// Mean and scale values obtained from the StandardScaler
const float mean[] = {6.76292482,   51.88701241,   81.17352496, 1009.03348166}; // Replace with actual mean values
const float scale[] = {4.00653756,  8.78442094, 14.9878748 ,  7.32198109}; // Replace with actual scale values

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

// Timing variables
unsigned long previousMillis = 0;
const long interval = 10000; // 60 seconds in milliseconds

void setup() {
  Serial.begin(9600);
  while (!Serial) {}

  // Initialize BME680
  if (!bme.begin()) {
    Serial.println("Could not find a valid BME680 sensor, check wiring!");
    while (1);
  }

  // Initialize BLE
  if (!BLE.begin()) {
    Serial.println("Starting BLE failed!");
    while (1);
  }

  BLE.setLocalName("WeatherPredictor");
  BLE.setAdvertisedService(weatherService);
  weatherService.addCharacteristic(weatherCharacteristic);
  BLE.addService(weatherService);
  weatherCharacteristic.writeValue("Weather data");

  BLE.advertise();
  Serial.println("BLE Weather Predictor device active, waiting for connections...");


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

// Function to convert float array to string
String floatArrayToString(float* array, int size) {
  String result = "";
  for (int i = 0; i < size; i++) {
    result += String(array[i]);
    if (i < size - 1) {
      result += ",";
    }
  }
  return result;
}

// Function to send input data via BLE
void sendInputDataBLE(float* inputData, int dataSize) {
  String dataString = floatArrayToString(inputData, dataSize);
  weatherCharacteristic.writeValue(dataString.c_str());
}

void makePrediction() {
  // Read sensor data
  if (!bme.performReading()) {
    Serial.println("Failed to perform reading from BME680 sensor!");
    return;
  }

  float temperature = ((bme.temperature * (9.0/5.0))+32)*(59.0/73.0);
  float pressure = (bme.pressure / 100.0); // hPa
  float humidity = (bme.humidity) * (82.0/55.0);
  
  Serial.print("Month = ");
  Serial.print(int(month));
  Serial.println("");

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
  float inputData[] = {month, temperature, humidity, pressure};
  // Send the input data via BLE
  sendInputDataBLE(inputData, 4);

  // Scale the input data
  scaleInputData(inputData, 4);

  // Prepare the input tensor
  tflInputTensor->data.f[0] = inputData[0];
  tflInputTensor->data.f[1] = inputData[1];
  tflInputTensor->data.f[2] = inputData[2];
  tflInputTensor->data.f[3] = inputData[3];

  // Run inference
  if (tflInterpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed");
    return;
  }

  // Get the prediction results from the output tensor
  float no_rain = tflOutputTensor->data.f[0];
  float rain = tflOutputTensor->data.f[1];

  // Determine the prediction label
  String prediction;
  if (no_rain > rain) {
    prediction = "No Rain in the next 30min";
  } else {
    prediction = "Rain in the next 30min";
  }

  float outputData[] = {no_rain, rain};
  // Print the results
  Serial.print("Prediction: "); Serial.println(floatArrayToString(outputData, 2));

  // Send the prediction via BLE
  weatherCharacteristic.writeValue(prediction.c_str());
}


void loop() {
  // wait for a Bluetooth® Low Energy central
  BLEDevice central = BLE.central();

  // if a central is connected to the peripheral:
  if (central) {
    Serial.print("Connected to central");
    // turn on the LED to indicate the connection:
    digitalWrite(LED_BUILTIN, HIGH);

    // check the battery level every 200ms
    // while the central is connected:
    while (central.connected()) {
      long currentMillis = millis();
      if (previousMillis==0 || (currentMillis - previousMillis >= interval)) {
        previousMillis = currentMillis;
        makePrediction();
  }
    }
    // when the central disconnects, turn off the LED:
    digitalWrite(LED_BUILTIN, LOW);
    Serial.print("Disconnected from central");
  }
}
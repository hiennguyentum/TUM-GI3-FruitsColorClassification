/** ------------------------------------------------------------------
 *  Important libraries  
 **/
#include <Wire.h>
#include "Adafruit_TCS34725.h"  // TCS34725 Grove Color Sensor library
#include <TensorFlowLite.h>     // TensorflowLite library
#include <CayenneLPP.h>         // Cayenne Library
#include <LoRaWan.h>            // Seeduino LoRaWAN library (also referred as "lora" below)
#include "model.h"              // Fruit classification tensorflowlite model


/** ------------------------------------------------------------------
 *  TensorflowLite libaries
 *  @ Arduino_TensorflowLite 1.14.0-ALPHA 
 **/ 
#include "tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"


/** ------------------------------------------------------------------
 *  TensorflowLite dependencies
 **/ 
// global variables used for TensorFlow Lite (Micro)
tflite::MicroErrorReporter tflErrorReporter;

// pull in all the TFLM ops, you can remove this line and
// only pull in the TFLM ops you need, if would like to reduce
// the compiled size of the sketch.
tflite::ops::micro::AllOpsResolver tflOpsResolver;

const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

// Create a static memory buffer for TFLM, the size may need to
// be adjusted based on the model you are using
constexpr int tensorArenaSize = 8 * 1024;
byte tensorArena[tensorArenaSize];


/** -------------------------------------------------------------------
 *  Other important dependencies
 **/
CayenneLPP lpp(51);   // Define CayenneLPP buffer size

// Grove_TCS32475 Color Sensor 
Adafruit_TCS34725 tcs = Adafruit_TCS34725(TCS34725_INTEGRATIONTIME_50MS, TCS34725_GAIN_4X);


/** -------------------------------------------------------------------
 *  Some nice bool*, char* and int* 
 **/
// Declare a string buffer for Cayenne
char buffer[256];           

// Array to map color index to a name
const char* CLASSES[] = 
{
  "Apple",
  "Empty",
  "Fig",
  "Lemon", 
  "Lime"
};

// Array to map color index to the classes' name
int CLASSES_INDEX[] =
{
  0, //"Apple",
  1, //"Empty",
  2, //"Fig",
  3, //"Lemon", 
  4, //"Lime"
};

/** -------------------------------------------------------------------
 *  Some nice #define
 **/
// Seeeduino LoRaWAN 
#define PIN_GROVE_POWER 38

// Insert LoRa keys here 
#define DevEUI "0026B2511F031053"
#define AppEUI "70B3D57ED00302CB"
#define AppKey "23B25633C587EAAE0274D8D1E0F80432"

// Number of predicted classes
#define NUM_CLASSES (sizeof(CLASSES) / sizeof(CLASSES[0]))


/** ------------------------------------------------------------------
 *  setup() function
 **/
void setup() 
{
  /** --------------------------------------------------------------------------------
   *  Seeeduino LoRaWAN & Grove TCS34725 Startup Section
   **/
  // Provide power to the 4 Grove connectors of the board
  digitalWrite(PIN_GROVE_POWER, HIGH); 

  // Switching the board to PowerSaver mode
  for(int i = 0; i < 26; i ++)          // Set all pins to HIGH to save power (reduces the
    {                                   // current drawn during deep sleep by around 0.7mA).
        if (i!=13) {                    // Don't switch on the onboard user LED (pin 13).
          pinMode(i, OUTPUT);
          digitalWrite(i, HIGH);
        }
    }  

  delay(5000);                        // Wait 5 secs after reset/booting to give time for potential upload
                                      // of a new sketch (sketches cannot be uploaded when in sleep mode)

  // Initialize serial connection  
  Serial.begin(9600);
  delay(500);
  
  // Wait for serial port to connect. Needed for native USB port only
  while (!Serial) {};  
  Serial.println("Seeeduino LoRaWAN board started!");

  // Starting connection with TCS32475 Color Sensor
  if (tcs.begin()) {
    Serial.println("Color Sensor TCS34725 Connected!");
    Serial.println();
  } else {
    Serial.println("No TCS34725 found ... check your connections");
    while (1); // halt!
  }


  /** --------------------------------------------------------------------------------
   *  TensorflowLite Setup Section
   **/
  // Get the TFL representation of the model byte array
  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) 
  {
    Serial.println("Model schema mismatch!");
    while (1);
  }

  // Create an interpreter to run the model
  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize, &tflErrorReporter);

  // Allocate memory for the model's input and output tensors
  tflInterpreter->AllocateTensors();

  // Get pointers for the model's input and output tensors
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);

  
  /** --------------------------------------------------------------------------------
   *  LoRaWAN & TTN Setup Section
   **/
  // Config LoRaWAN
  lora.init();
  Serial.println("Connecting to 'Geoinformatik3 TTN LoRaWAN' ....... ");
  Serial.println();

  // Checking LoRaWAN version and id
  memset(buffer, 0, 256);
  lora.getVersion(buffer, 256, 1);
  if (Serial) {
    Serial.print(buffer);
  }
  memset(buffer, 0, 256);
  lora.getId(buffer, 256, 1);
  if (Serial) {
    Serial.print(buffer);
  }
  
  // void setId(char *DevAddr, char *DevEUI, char *AppEUI);
  lora.setId(NULL, DevEUI, AppEUI);  

  // setKey(char *NwkSKey, char *AppSKey, char *AppKey);
  lora.setKey(NULL, NULL, AppKey);
    
  lora.setDeciveMode(LWOTAA);           // select OTAA join mode (note that setDeciveMode is not a typo; it is misspelled in the library)
  // lora.setDataRate(DR5, EU868);         // SF7, 125 kbps (highest data rate)
  lora.setDataRate(DR3, EU868);         // SF9, 125 kbps (medium data rate and range)
  // lora.setDataRate(DR0, EU868);         // SF12, 125 kbps (lowest data rate, highest max. distance)

  // lora.setAdaptiveDataRate(false);  
  lora.setAdaptiveDataRate(true);       // automatically adapt the data rate
    
  lora.setChannel(0, 868.1);
  lora.setChannel(1, 868.3);
  lora.setChannel(2, 868.5);
  lora.setChannel(3, 867.1);
  lora.setChannel(4, 867.3);
  lora.setChannel(5, 867.5);
  lora.setChannel(6, 867.7);
  lora.setChannel(7, 867.9);

  lora.setDutyCycle(false);             // for debugging purposes only - should normally be activated
  lora.setJoinDutyCycle(false);         // for debugging purposes only - should normally be activated
    
  lora.setPower(14);                    // LoRa transceiver power (14 is the maximum for the 868 MHz band)
  lora.setPort(33);
 
  unsigned int nretries;
  nretries = 0;
  while (!lora.setOTAAJoin(JOIN, 20)) {
    nretries++;
    if (Serial) {
      Serial.println((String)"Join failed, retry: " + nretries);
    }
  }
  Serial.println();
  Serial.println("Join Geoinformatik3 TTN LoRaWAN successful!");
  Serial.println();


  /** --------------------------------------------------------------------------------
   *  Next Instruction
   **/
  Serial.println("Type 'help' in the serial command box for the command options");
  Serial.println();
}


/** ------------------------------------------------------------------
 *  loop() function
 **/
void loop() 
{
  if (!Serial.available()) 
  {
    delay(100);
    return;
  }

  // Create a string command
  String command = Serial.readStringUntil('\n');


    /** --------------------------------------------------------------------------------
    *  'Help' Command
    **/
    if (command == "help") 
    {
        Serial.println("Input command: 'help' registered ! ");       
        Serial.println();
        
        Serial.println("Available commands: ");
        Serial.println("\tcapture: to start capturing RGB color data of a new sample ");
        Serial.println("\tpredict: to classify a new sample using the model.h deployed on the board ");
        Serial.println("\tsend: to send current RGB color data to FROST Server via LoRaWAN ");
        Serial.println();
    }


    /** --------------------------------------------------------------------------------
    *  'Capture' Command
    **/
    else if (command == "capture") 
    {
        Serial.println("'capture' command registered ! ");       
        Serial.println();
          
        String objName;
        int numSamples;
      
        // Quick round of questionaires
        Serial.println("What is the name of the object that would you like to record? ");
        objName = readSerialString();
        Serial.println("How many samples would you like to record? ");
        numSamples = readSerialNumber();
        Serial.println();

        Serial.println((String)"Capturing " + "'" + numSamples + "'" + " samples of " + "[" + objName + "] ...........");
        Serial.println("*2 seconds delay is applied between each record for rotating the object* ");
        Serial.println();
        
        // Print the header
        Serial.println("Red,Green,Blue");        
        for (int i = 0; i < numSamples; i++) 
        {
            float red, green, blue;                         // Float values for RGB data 
            readColorSensorData(&red, &green, &blue);       // Getting RGB data from sensor
      
            // Print the data in CSV format
            Serial.println((String) red + "," + green + "," + blue);
            delay(1000); // Give users 2 seconds to rotate object's surface
        }
        Serial.println("Done");
        Serial.println();

        Serial.println("Type in the command box either 'capture', 'predict' or 'send' to continue ");       
        Serial.println();
    }


    /** --------------------------------------------------------------------------------
    *  'Predict' Command
    **/
    else if (command == "predict") 
    {
        Serial.println("'predict' command registered ! ");       
        Serial.println();
        
        float red, green, blue;                         // Float values for RGB data 
        readColorSensorData(&red, &green, &blue);       // Getting RGB data from sensor

        // Input current sensor reading data to tensorflow model
        tflInputTensor->data.f[0] = red;
        tflInputTensor->data.f[1] = green;
        tflInputTensor->data.f[2] = blue;

        // Run inferencing
        TfLiteStatus invokeStatus = tflInterpreter->Invoke();
        
        if (invokeStatus != kTfLiteOk) 
        {
          Serial.println("Invoke failed!");
          while (1);
          return;
        }

        // Print the current RGB values
        Serial.println("Current RGB values from the sensor: ");
        Serial.println((String) "R: " + red + " | G: " + green + " | B: " + blue);
        Serial.println();
        Serial.println("Prediction from the above RGB values are: ");

        for (int i = 0; i < NUM_CLASSES; i++) 
        {
          int probability[] = {tflOutputTensor->data.f[i] * 100};
          Serial.println(probability[i]);
        }

        // Output the prediction
        for (int i = 0; i < NUM_CLASSES; i++) 
        {
          Serial.print(CLASSES[i]);
          Serial.print("[");
          Serial.print(CLASSES_INDEX[i]);
          Serial.print("]");
          Serial.print(": ");
          Serial.print(int(tflOutputTensor->data.f[i] * 100)); 
          Serial.print("%\n");
        }

        Serial.println();
        Serial.println("Type in the command box either 'capture', 'predict' or 'send' to continue ");       
        Serial.println();
    }


    /** --------------------------------------------------------------------------------
    *  'Send' Command
    **/
    else if (command == "send") 
    {
        Serial.println("'send' command registered ! ");       
        Serial.println();
        
        bool result = false;                            // Boolean for LoRa package transferring
        unsigned int nloops = 0;                        // Loop counting for sending LoRa package      
        float red, green, blue;                         // Float values for RGB data 
        readColorSensorData(&red, &green, &blue);       // Getting RGB data from sensor

        // Input current sensor reading data to tensorflow model
        tflInputTensor->data.f[0] = red;
        tflInputTensor->data.f[1] = green;
        tflInputTensor->data.f[2] = blue;

        // Run inferencing
        TfLiteStatus invokeStatus = tflInterpreter->Invoke();
        
        if (invokeStatus != kTfLiteOk) 
        {
          Serial.println("Invoke failed!");
          while (1);
          return;
        }
        
        // Print the current RGB values
        Serial.println("Current RGB values from the sensor are: ");
        Serial.println((String) "R: " + red + " | G: " + green + " | B: " + blue);
        Serial.println();
        Serial.println("Prediction from the current RGB values are: ");

        // Output the prediction
        for (int i = 0; i < NUM_CLASSES; i++) 
        {
          Serial.print(CLASSES[i]); 
          Serial.print(" = ");
          Serial.print(int(tflOutputTensor->data.f[i] * 100)); // 
          Serial.print("%\n");
        }

        lpp.reset();                           // Resets the Cayenne buffer
        lpp.addAnalogOutput(1, red);          // Encodes the red value as float on channel 1 in Cayenne AnalogOutput format 
        lpp.addAnalogOutput(2, green);        // Encodes the green value as float on channel 2 in Cayenne AnalogOutput format 
        lpp.addAnalogOutput(3, blue);         // Encodes the blue value as float on channel 3 in Cayenne AnalogOutput format 
        
        // Checking the data sending loop
        nloops++;
        if (Serial) 
        {
          Serial.println();
          Serial.println((String)"Sending package #: " + nloops + " ......");
          Serial.println();
        }
        
        // Send it off
        result = lora.transferPacket(lpp.getBuffer(), lpp.getSize(), 5);   // sends the Cayenne encoded data packet (n bytes) with a default timeout of 5 secs
        // result = lora.transferPacket(lpp.getBuffer(), 5);
          
        if (result)
        {
          char rx[256];
          short length;
          short rssi;
          length = lora.receivePacket(rx, 256, &rssi);
          
          if (length)
          {
            if (Serial) 
            {
              Serial.println((String)"Length is: " + length);
              Serial.println((String)"RSSI is: " + rssi);
              Serial.println("Data is: ");
              for (unsigned char i = 0; i < length; i ++)
              {
                Serial.print("0x");
                Serial.print(rx[i], HEX);
                Serial.print(" "); 
              }        
              
              // Convert received package to int
              int rx_data_asInteger = atoi(rx);      
              Serial.println("Received data: " + String(rx_data_asInteger));
            }
          } 
         }
  
        if (Serial) 
        {
          Serial.println();
          Serial.println((String)"Package # " + nloops + " sent succesfully!\n");
        }
 
//        delay(60000);
        
        Serial.println("Type in the command box either 'capture', 'predict' or 'send' to continue ");       
        Serial.println();
    }
}



/** ------------------------------------------------------------------
 *  Read input number input in the debugSerial command
 *  @return
 **/
int readSerialNumber() 
{
    while (!Serial.available()) delay(1);

    return Serial.readStringUntil('\n').toInt();
}



/** ------------------------------------------------------------------
 *  Read input number input in the debugSerial command
 *  @return
 **/
String readSerialString() 
{
    
    while (!Serial.available()) delay(1);

    return Serial.readStringUntil('\n');
}



/** ------------------------------------------------------------------
 * Function to help read color sensor data and return as RGB
 **/
void readColorSensorData(float *r, float *g, float *b)
{
  // Define RGB and Sum (for CSV) float
  float red, green, blue;

  // Read the color data
  tcs.setInterrupt(false);  // turn on LED
  delay(60);  // takes 60ms to read
  tcs.getRGB(&red, &green, &blue);
  tcs.setInterrupt(true);  // turn off LED

  *r = red;
  *g = green;
  *b = blue;

  return;
}

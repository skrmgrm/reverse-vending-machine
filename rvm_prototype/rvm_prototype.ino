#include <Servo.h>
#include "HX711.h"

#define calibration_factor -7050.0  //This value is obtained using the SparkFun_HX711_Calibration sketch

#define LOADCELL_DOUT_PIN 5
#define LOADCELL_SCK_PIN 4

#define proximity1 A10
#define proximity2 A3
#define proximity3 A0

#define redLed 8
#define greenLed 9

#define objectDetected proximityStatus1 == 0 || proximityStatus2 == 0 || proximityStatus3 == 0

Servo entranceServo;
Servo sorterServo;

HX711 scale;

void setup() {
  // put your setup code here, to run once:


  entranceServo.attach(24);
  sorterServo.attach(22);
  entranceServo.write(65);
  delay(1000);
  sorterServo.write(75);
  delay(1000);

  pinMode(redLed, OUTPUT);
  pinMode(greenLed, OUTPUT);

  

  pinMode(proximity1, INPUT);
  pinMode(proximity2, INPUT);
  pinMode(proximity3, INPUT);

  Serial.println("HX711 scale demo");
  scale.begin(LOADCELL_DOUT_PIN, LOADCELL_SCK_PIN);
  scale.set_scale(calibration_factor);  //This value is obtained by using the SparkFun_HX711_Calibration sketch
  scale.tare();                         //Assuming there is no weight on the scale at start up, reset the scale to 0
}

void loop() {
  // put your main code here, to run repeatedly:
  static char serial;

  int proximityStatus1 = digitalRead(proximity1);
  int proximityStatus2 = digitalRead(proximity2);
  int proximityStatus3 = digitalRead(proximity3);
  
  digitalWrite(greenLed, HIGH);

  if (objectDetected) {
 
    Serial.print("Reading: ");

    float weight = scale.get_units();
    Serial.print(weight);  //scale.get_units() returns a float
    Serial.print(" g");    //You can change this to kg but you'll need to refactor the calibration_factor
    Serial.println();

    delay(1500);

    if (weight < 3.00) {
      entranceServo.write(150);  // sets the servo position according to the scaled value
      delay(2000);
      entranceServo.write(65);  // sets the servo position according to the scaled value
      Serial.begin(9600);
      delay(1500);
      if(Serial.available()) {
        serial = Serial.read();
        
        if (serial == '2') {
          objectAccepted();
        }
        if(serial == '1') {
          objectRejected();
        }
      }

      
    
    } else {
      objectRejected();
    }
  }

  // entranceServo.write(150);  // sets the servo position according to the scaled value
  // delay(2000);
  // entranceServo.write(65);  // sets the servo position according to the scaled value
  // delay(2000);


  // sorterServo.write(20);  // sets the servo position according to the scaled value
  // delay(2000);
  // sorterServo.write(75);  // sets the servo position according to the scaled value
  // delay(2000);
  // sorterServo.write(150);  // sets the servo position according to the scaled value
  // delay(2000);
  // sorterServo.write(75);  // sets the servo position according to the scaled value
  // delay(2000);
}

void objectAccepted() {
  Serial.println("Object Accepted");
  digitalWrite(greenLed, LOW);
  
  sorterServo.write(150);
  delay(2000);
  sorterServo.write(75);
  delay(2000);
  digitalWrite(greenLed, HIGH);
  Serial.end();
}

void objectRejected() {
  Serial.println("Object Rejected");
  digitalWrite(redLed, HIGH);
  sorterServo.write(20);
  delay(2000);
  sorterServo.write(75);
  delay(2000);
 
  digitalWrite(redLed, LOW);
  Serial.end();
}

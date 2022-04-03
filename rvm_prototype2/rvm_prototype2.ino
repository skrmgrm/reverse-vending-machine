#include <Servo.h>
#include "HX711.h"

#define calibration_factor -7050.0  //This value is obtained using the SparkFun_HX711_Calibration sketch

#define LOADCELL_DOUT_PIN 5
#define LOADCELL_SCK_PIN 4

#define redLed 8
#define greenLed 9

int bottleNumbers = 0;

Servo entranceServo;
Servo sorterServo;
Servo dispenserServo;

HX711 scale;

void setup() {
  // put your setup code here, to run once:

  servoSetup();
  pinMode(redLed, OUTPUT);
  pinMode(greenLed, OUTPUT);

  Serial.println("HX711 scale demo");
  scale.begin(LOADCELL_DOUT_PIN, LOADCELL_SCK_PIN);
  scale.set_scale(calibration_factor);  //This value is obtained by using the SparkFun_HX711_Calibration sketch
  scale.tare();                         //Assuming there is no weight on the scale at start up, reset the scale to 0
}

void loop() {
  // put your main code here, to run repeatedly:
  static char serial;

  Serial.print("Reading: ");

  float weight = scale.get_units();
  weight = DecimalRound(weight, 1);
  Serial.print(weight);  //scale.get_units() returns a float
  Serial.print(" g");    //You can change this to kg but you'll need to refactor the calibration_factor
  Serial.println();

  digitalWrite(greenLed, HIGH);

  if (weight > 0) {
    if (weight < 3.00) {
      delay(1000);
      Serial.begin(9600);
      if (Serial.available()) {
        serial = Serial.read();

        if (serial == '2') {
          objectAccepted();
        } else if (serial == '1') {
          objectRejected();
        }
      }
    }
    else if ( weight > 3.00)  {
      Serial.println("Object Rejected");
      objectRejected();
    }
  }
}

float DecimalRound(float input, int decimals)
{
  float scale = pow(10, decimals);
  return round(input * scale) / scale;
}

void objectAccepted() {
  Serial.println("Object Accepted");
  digitalWrite(greenLed, LOW);
  bottleNumbers += 1;
  entranceServo.write(150);  // sets the servo position according to the scaled value
  delay(2000);
  sorterServo.write(150);
  delay(2000);
  sorterServo.write(75);
  delay(2000);
  entranceServo.write(65);  // sets the servo position according to the scaled value
  delay(1500);
  digitalWrite(greenLed, HIGH);
  Serial.end();
}

void objectRejected() {
  Serial.println("Object Rejected");
  digitalWrite(redLed, HIGH);
  entranceServo.write(150);  // sets the servo position according to the scaled value
  delay(2000);
  sorterServo.write(20);
  delay(2000);
  sorterServo.write(75);
  delay(2000);
  entranceServo.write(65);  // sets the servo position according to the scaled value
  delay(1500);
  digitalWrite(redLed, LOW);
  Serial.end();
}

void servoSetup() {
  entranceServo.attach(24);
  sorterServo.attach(22);
  dispenserServo.attach(23);
  delay(1000);
  sorterServo.write(75);
  delay(1000);
  entranceServo.write(65);
  delay(1000);
}

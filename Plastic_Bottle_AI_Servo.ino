
#include <Servo.h>

Servo platformServo;
#define platformServoPin 9
#define IRSensor 2



void setup() {
  Serial.begin(9600);

  //  IRsensor
  pinMode(IRSensor, INPUT);

  // initialize servo
  platformServo.attach(platformServoPin);
  platformServo.write(0);
  delay(1000);
  platformServo.write(90);
  delay(1000);
  platformServo.write(180);
  delay(1000);
  platformServo.write(90);
  delay(1000);
}

void loop() {
  static char serialOutput;
  int statusSensor = digitalRead(IRSensor);


  while (Serial.available()) {
    serialOutput = Serial.read();

  }

  if (serialOutput == '2' && statusSensor == 0) {
    delay(1000);
    platformServo.write(180);
    delay(1000);
  
  } else if (serialOutput == '1' && statusSensor == 0) {
    delay(1000);
    platformServo.write(0);
    delay(1000);
 
  } 

  if(statusSensor == 1) {
    platformServo.write(90);
  }

  //  if(statusSensor == 0){
  //    Serial.begin(9600);
  //  }else if(statusSensor == 1){
  //    Serial.end();
  //    platformServo.write(90);
  //  }

}

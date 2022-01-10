#include <HX711.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <Keypad.h>
#include <Servo.h>

#define calibration_factor -7050 //This value is obtained using the SparkFun_HX711_Calibration sketch
#define LOADCELL_DOUT_PIN  3
#define LOADCELL_SCK_PIN  2

#define ESPin 53
#define SSPin 51
#define IRSensor 4
#define buttonPin 8

int buttonState = 0;
int buttonLastState = 0;
int voucherState = 0;

// voucher code
int bottleCount = 0;
int i = 0;
int j = 0;

String letters[] = {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"};
String randString = "";

Servo EServo;
Servo SServo;

LiquidCrystal_I2C lcd(0x27, 16, 2);

HX711 scale;

void setup() {

  pinMode(IRSensor, INPUT);
  pinMode(buttonPin, INPUT_PULLUP);
  delay(1000);

  initialize();
}

void loop() {
  static char serialOutput;
  int statusSensor = digitalRead(IRSensor);
  float weightSensor = scale.get_units();

//  lcd.setCursor(0, 0);
//  lcd.println(scale.get_units(), 1); //scale.get_units() returns a float
//  lcd.setCursor(0, 1);
//  lcd.print(" g"); //You can change this to kg but you'll need to refactor the calibration_factor


  buttonState = digitalRead(buttonPin);
  Serial.println(buttonState);

  buttonLastState = buttonState;

  if (buttonState == 1 && buttonLastState == 1) {
    if (voucherState == 0) {
      calculateVoucher();
      voucherState = 1;
    } else if (voucherState == 1) {
      reset();
      voucherState = 0;
    }
  }

  if (voucherState == 1) {
    releaseVoucher();
  }


  if (statusSensor == 0 && weightSensor < 2.00) {
    delay(500);
    Serial.begin(9600);
    serialOutput = Serial.read();
    if (serialOutput == '2') {
      acceptObject();
    } else if (serialOutput == '1') {
      rejectObject();
    }
  } else if (statusSensor == 0 && weightSensor > 3.00) {
    rejectObject();
  }
  else if (statusSensor == 1) {
    standby();
  } else {
    rejectObject();
  }
}

void acceptObject() {
  SServo.attach(SSPin);
  SServo.write(130);
  delay(1500);
  SServo.detach();
  EServo.attach(ESPin);
  EServo.write(200);
  delay(1500);
  EServo.detach();

  Serial.end();
  bottleCount += 1;


}

void rejectObject() {
  SServo.attach(SSPin);
  SServo.write(80);
  delay(1500);
  SServo.detach();
  EServo.attach(ESPin);
  EServo.write(200);
  delay(1500);
  EServo.detach();

  Serial.end();
}

void standby() {
  EServo.attach(ESPin);
  EServo.write(85);
  delay(1500);
  EServo.detach();
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Bottle Count: ");
  lcd.setCursor(0, 1);
  lcd.print(bottleCount);
}

void initialize() {
  lcd.begin();
  lcd.clear();
  lcd.backlight();
  lcd.setCursor(0, 0);
  lcd.print("Starting...");
  delay(1000);
  lcd.clear();

  scale.begin(LOADCELL_DOUT_PIN, LOADCELL_SCK_PIN);
  scale.set_scale(calibration_factor); //This value is obtained by using the SparkFun_HX711_Calibration sketch
  scale.tare(); //Assuming there is no weight on the scale at start up, reset the scale to 0

}

void calculateVoucher() {
  lcd.clear();

  for (i = 0; i < 8; i++)
  {
    randString = randString + letters[random(0, 25)];
  }

  randString += String(bottleCount);

}

void releaseVoucher() {

  lcd.setCursor(0, 0);
  lcd.print("Voucher Code");
  lcd.setCursor(0, 1);
  lcd.print(randString);
}

void reset() {
  randString = "";
  bottleCount = 0;
}

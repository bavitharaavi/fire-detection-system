/*
 * Fire detection sensor node (Arduino Uno / Nano / ESP32)
 *
 * Sensors
 *   MQ-2 smoke/gas sensor   -> A0  (analog)
 *   IR flame sensor (KY-026) -> D2  (digital, LOW when flame detected on most modules)
 *   DHT22 temperature        -> D4
 *   Active buzzer            -> D8
 *
 * Output: one JSON line every 500 ms on Serial @ 9600 baud
 *   {"temp_c":24.6,"smoke_ppm":132,"flame":0}
 * Input commands (newline terminated): BUZZER_ON | BUZZER_OFF
 *
 * Library: "DHT sensor library" by Adafruit (Library Manager).
 */
#include <DHT.h>

#define MQ2_PIN      A0
#define FLAME_PIN    2
#define DHT_PIN      4
#define BUZZER_PIN   8
#define DHT_TYPE     DHT22

// MQ-2 calibration: R0 is the sensor resistance in clean air (run calibrateR0()
// once after a 24-48 h burn-in and paste the printed value here).
const float RL_KOHM = 10.0;
float R0_KOHM = 9.83;

DHT dht(DHT_PIN, DHT_TYPE);
unsigned long lastSend = 0;
String cmd;

float readRsKohm() {
  int raw = analogRead(MQ2_PIN);
  float v = raw * (5.0 / 1023.0);
  if (v < 0.01) v = 0.01;
  return (5.0 - v) / v * RL_KOHM;
}

// Approximate smoke ppm from the MQ-2 datasheet curve: log(ppm) = (log(Rs/R0) - b) / m
float smokePpm() {
  const float m = -0.44, b = 1.62;
  float ratio = readRsKohm() / R0_KOHM;
  return pow(10, (log10(ratio) - b) / m);
}

void calibrateR0() {
  float sum = 0;
  for (int i = 0; i < 50; i++) { sum += readRsKohm(); delay(100); }
  Serial.print("{\"r0_kohm\":"); Serial.print(sum / 50.0 / 9.83, 3); Serial.println("}");
}

void setup() {
  Serial.begin(9600);
  pinMode(FLAME_PIN, INPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);
  dht.begin();
}

void handleCommands() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      cmd.trim();
      if (cmd == "BUZZER_ON")  digitalWrite(BUZZER_PIN, HIGH);
      if (cmd == "BUZZER_OFF") digitalWrite(BUZZER_PIN, LOW);
      if (cmd == "CALIBRATE")  calibrateR0();
      cmd = "";
    } else {
      cmd += c;
    }
  }
}

void loop() {
  handleCommands();
  if (millis() - lastSend < 500) return;
  lastSend = millis();

  float t = dht.readTemperature();
  int flame = digitalRead(FLAME_PIN) == LOW ? 1 : 0;

  Serial.print("{\"temp_c\":");
  if (isnan(t)) Serial.print("null"); else Serial.print(t, 1);
  Serial.print(",\"smoke_ppm\":");
  Serial.print(smokePpm(), 0);
  Serial.print(",\"flame\":");
  Serial.print(flame);
  Serial.println("}");
}

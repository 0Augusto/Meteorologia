#include "DHT.h"

#define DHTPIN A1      // Pino conectado ao sensor
#define DHTTYPE DHT11  // Tipo do sensor

DHT dht(DHTPIN, DHTTYPE);

void setup() 
{
  Serial.begin(9600);
  Serial.println("Inicializando sensor DHT...");
  dht.begin();
}

void loop() 
{
  delay(5000); // Aguarda 5 segundos para nova leitura

  float humidity = dht.readHumidity();
  float temperature = dht.readTemperature();

  if (isnan(humidity) || isnan(temperature)) 
  {
    Serial.println("Erro ao ler o sensor!");
    return;
  }

  Serial.print("Umidade: ");
  Serial.print(humidity);
  Serial.print(" %\t");
  Serial.print("Temperatura: ");
  Serial.print(temperature);
  Serial.println(" °C");
}

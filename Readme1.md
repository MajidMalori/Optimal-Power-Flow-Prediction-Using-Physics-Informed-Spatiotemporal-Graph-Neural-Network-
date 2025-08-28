# Remote Soil Resitance Monitoring and Alerting System using GSM
![Main Image](images/main-image.jpg)

The Remote Soil Resistance Monitoring and Alerting System is an experimental 'mini' project built to explore and harness the power of sensor-based IOT systems. The two main parameters that determine soil resistivity, soil moisture content and soil temperature, were used to estimate the soil resistance, a crucial factor for electrical safety (home grounding) and even agriculture (improved irrigation efficiency - up to 50% water savings - and crop yield increases of 10-30%). The soil resistance is continuously monitored in real-time, with results sent via GSM technology directly to an operator’s mobile phone. While the current system provides estimates, ongoing development strives for improved accuracy. It's already proving valuable for technicians, with potential future applications for Ghanaian farmers.


## Acknowledgements
I learned about the challenges of manual soil resistance checks for PV array grounding using testers like the Fluke 1625. These checks are vital for safety in vast mines, but they're time-consuming and laborious. Inspired by a Coursera course on [Internet of Things (IoT): Sensing and Actuation From Devices](https://www.coursera.org/learn/internet-of-things-sensing-actuation) taught by Professor Harinath Garudadri at the University of California San Diego, my friend Lemuel and I designed a miniaturized soil resistance meter to offer a solution.


## Authors

- Bernard : [@bengentle10](https://www.github.com/bengentle10)
- Lemuel

## Things used in this project
### Hardware Components
- [SIM900 GSM/GPRS](https://lastminuteengineers.com/sim900-gsm-shield-arduino-tutorial/)
- Arduino Uno R3
- ATMEGA328P-PU Microcontroller (PDIP)
- 16Mhz Crystal oscillato
- DS18B20 Water Proof Temperature Probe
- Soil Moisture Sensor Module
- 16X2 LCD Display (Green) with I2C Backpack 5V
- LM2596 DC to DC Buck Converter
- 0.1uF 50v Electrolytic Capacitor (10pcs)
- 10uF 50v Electrolytic Capacitor (10pcs)
- 5v Voltage regulator-LM7805
- BC547B NPN Transistor (10pcs)
- 10k ohms resistor (10pc)
- 2-Pin terminal blocks
- LED – Red 5mm (10pcs)
- 18650 2000mAh Lithium Batteries
- Lithium battery charger
- Active Buzzer
- Breadboard

### Software Apps and Online Services
- [Arduino IDE](https://www.arduino.cc/en/software)
- [Proteus Software](https://www.edgeimpulse.com/). Before building the real circuit, Proteus software allowed me to simulate the soil resistance meter's behavior. This virtual breadboard environment let me test circuit functionality and refine the design using a model for soil resistance under various conditions. Essentially, Proteus helped ensure the circuit would work properly with real soil before any physical components were needed.
- Fusion 360

## Methodology
### Remote Soil Resistance Monitoring and Alerting System
![Schematic](images/schematic.png)
   - **Sensor Data Acquisition**: The code reads temperature and moisture sensor values.
   - **Soil Resistance Calculation**: Based on the acquired data, the code computes the soil resistance value.
   - **Threshold Monitoring**: A predefined threshold value of 5 ohms is set. If the calculated resistance exceeds this value, indicating a potential issue, an alarm is triggered.
  - **Alert System**: Upon exceeding the threshold, a buzzer sounds, and the GSM module is activated.
  - **Data Transmission**: The GSM module transmits the collected sensor data (temperature, moisture, and calculated resistance) to a designated mobile phone number.

### Proteus Simulation
1. ***Proteus Software Setup:*** After installing Proteus software, you'll need to download and import additional libraries for this project. Proteus doesn't include these libraries by default. A list of the required libraries and a download link can be found below
   - [Arduino library for proteus](http://www.theengineeringprojects.com/2015/12/arduino-library-proteus-simulation.html)
   - [GSM library for proteus](http://www.theengineeringprojects.com/2016/03/gsm-library-proteus.html)
   - [Buzzer library for proteus](https://github.com/satyamkr80/Buzzer-library-for-proteus)

2. ***Proteus Simulation:*** Prior to physical implementation, the electrical schematic for the soil resistance system was first developed within the Proteus software environment, as detailed in the following explanation and diagram.

   1. Power Circuit

     - Located in the upper left corner of the circuit.
     - Supplies 5V power to the microcontroller (U1) using a 36 ATMEGA 382 chip.
     - Utilizes a crystal (X1) to provide a clock signal for the microcontroller, ensuring it remains active.
     - Capacitors (C5 and C6) reduce noise in the clock signal for stable operation.
     - Resistor (R1) limits current flow to the microcontroller, protecting it from damage.
  
   2. Sensor Inputs

    - Represented by RV1 (moisture sensor) and U3 (temperature sensor).
     - These sensors measure soil moisture and temperature in the field.
     - The measured values are sent to the microcontroller as input data.
  
   3. Serial Communication Ports (RXD and TXD)
    
     - These ports allow connection of a GSM modem.
     - The GSM modem receives information from the microcontroller's output (node 10).
  
  
![Circuit Design in Proteus](images/simulation.png)

3. ***Programming the Microcontroller:*** The ATmega328P microcontroller from the Proteus software environment was programmed using the [Arduino IDE](https://support.arduino.cc/hc/en-us/articles/9207690465436-IDE-Support) to process sensor data and trigger alerts. The repository contains all the resources you need to program your ATmega328 microcontroller. It includes two Arduino code files: "Arduino code - Proteus Simulation.ino" for simulating the program in Proteus software and "Arduino Code - Physical Implementation.ino" for running the program on the actual hardware. Additionally, a file named "Proteus Simulation.pdsprj" is provided for simulating the circuit in Proteus, allowing you to test your program in a virtual environment before implementing it on the physical ATmega328.


After setting up the circuit design and programming the microcontroller, the Proteus simulation successfully replicated the designed functionality. The system accurately calculated and displayed soil resistance on the LCD [top right corner] based on simulated temperature and moisture sensor values. Additionally, it demonstrated remote data transmission capabilities, as evidenced by sensor data displayed in the console [middle left corner]. See the image below for a visualization of these outputs.

**Results for Fixed Temperature of 3&deg;C and Moisture Content of 6%**

![Simulation Output](images/gsm-simulation.png)


### Physical Implementation

1. ***Testing of Sensors***: The functionality of the temperature and soil moisture sensors was verified by collecting data to calculate soil resistance. The calculated value was then displayed on the LCD. The output of this test is shown below

![Sensor Testing](images/lcd-display.png)
   
2. ***ATmega328 configured as Arduino Uno (external programmer)***:The Arduino Uno is a great choice for beginners in microcontroller programming because of its easy-to-use interface and pre-built components. This makes uploading code to the ATmega328 microcontroller on the Arduino Uno much simpler than using a standalone ATmega328 chip, which requires extra circuitry and a separate programmer to set up. It's important to consider the size of the Arduino Uno, though, as it might not fit well in projects with limited space, like the final 3D-printed meter housing for this project.To achieve a more compact design, the ATmega328 microcontroller was selected. Leveraging the Arduino IDE and an external programmer (or an Arduino Uno configured as an In-System Programmer), the code was uploaded directly to the ATmega328. A dedicated circuit was designed and implemented to provide the necessary functionality. Refer to the provided link [here](https://www.instructables.com/Programming-ATmega328-With-Arduino-IDE-Using-8MHz-/) for detailed instructions on this configuration process.
   
*An already configured Atmega328*
![circuitary](images/circuit.png)

3. ***3D-printed circuit housing***: After the circuit design, a custom enclosure was designed using Fusion 360 (3D modeling software) and subsequently fabricated using a 3D printer. This enclosure serves a critical purpose: to protect the delicate circuitry from dust, moisture, and potential physical damage. The finished 3D-printed housing is shown below.
   
![housing-modeling](images/housing-modeling.jpg)


4. ***Powering the Project ( lithium-ion batteries)***: To ensure reliable operation, a custom power supply solution was developed for the project. This design utilizes four 18650 2000mAh lithium-ion batteries, chosen based on a careful analysis of the overall circuit's power requirements and the available space within the 3D-printed housing. These batteries provide ample capacity to power the ATmega328 microcontroller, supporting circuitry, sensors, buzzer, and GSM module. The batteries are configured in a combination of parallel and series connections. This approach aims to double both the current and voltage output. This is necessary because certain components, particularly the GSM module, have a higher current draw. 

  - **Protecting the ATmega328**: The ATmega328 microcontroller and its supporting circuitry can only handle a maximum voltage of 5 volts, hence a voltage regulator is essential to prevent damage. Two options are available:
     - *Buck Converter*: A buck converter can be used to efficiently step down the combined voltage from the battery configuration to a safe level for the microcontroller (typically 5 volts).
     - *5V Voltage Regulator (LM7805)*: Alternatively, a simpler approach is to employ a dedicated 5V voltage regulator like the LM7805. This component ensures a consistent 5V supply to the microcontroller.
 - **GSM Module Power Requirements**: The GSM module, on the other hand, typically operates within a voltage range of 3.4V to 4.4V and can draw a maximum current of 2 amps. Depending on the specific module used, a separate buck converter might be necessary to provide the appropriate voltage range for its operation.
  
   
5. ***GSM model (SIM900) connection with Microcontroller***: The project incorporates a GSM module to facilitate wireless data transmission. This module requires a standard SIM card (often referred to as a "large" SIM card) with a sufficient data plan to enable communication. The choice of network provider is user-dependent, allowing for flexibility based on individual needs and coverage areas.For the ATmega328 microcontroller to interact with the GSM module and transmit data, it's crucial to program the microcontroller with the specific phone number associated with the inserted SIM card. This phone number serves as the designated recipient for data alerts generated by the soil resistance meter. For detailed instructions on connecting a GSM module to a microcontroller, refer to the provided link: [click here](https://lastminuteengineers.com/sim900-gsm-shield-arduino-tutorial/).

GSM circuit connection with microcontroller
   
![GSM](images/GSM.jpg)

After completing the development of each phase of the project, the components were assembled and housed for comprehensive testing.
  
## Result and Conclusion

Our device captured real-time soil data for both dry and moist conditions. When the soil was dry, it registered a temperature of 32°C, a moisture content of only 2%, and a much higher resistivity of 2.08 ohm-meters (Ωm). In contrast, moist soil displayed a cooler temperature, a much higher moisture content (exact percentage not specified), and a significantly lower resistivity of just 0.20 Ωm. This test confirms what we already know: drier soil holds less moisture and conducts electricity less efficiently, resulting in a higher resistivity reading. Conversely, moist soil conducts electricity much better, with a lower resistivity value. 

The initial design of the remote soil resistivity monitoring system showed promise. It successfully functioned as expected during both the simulations conducted with Proteus version 8.6 software as well as the physical implementation. The system's remote monitoring capability allows for timely intervention in situations where soil resistivity deviates from desired parameters. Upon detecting such conditions, an SMS notification would be triggered, alerting designated personnel to take corrective actions and improve soil conditions.

![GSM SMS](images/GSM%20SMS.jpg)

While the initial design of the remote soil resistivity monitoring system successfully utilized common components and functioned well in simulations, future advancements can broaden its impact beyond electrical applications.  This system has the potential to empower not only electrical companies and engineers with real-time data and automated alerts for improved soil monitoring in earthing applications, but also agriculturists. By understanding soil resistivity, farmers can gain valuable insights into soil health -  higher resistivity often indicates lower moisture content. This information can be used to optimize irrigation practices, potentially saving water and improving crop yields. Additionally, exploring cost-effective data transmission alternatives like Wi-Fi or LoRaWAN could make the system more affordable for wider adoption.  Most importantly, this design offers a chance to revolutionize soil resistivity monitoring by replacing manual methods with a time-saving and cost-effective solution, making it accessible for various sectors, even households

A brief demonstration of the project is shown below:

![Demonstration](images/demonstration.gif)

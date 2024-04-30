# WeatherPatrol
EE 400A: TinyML Project Documentation

Proposal Summary:

**Background**: The Pacific Northwest is a geographic spectacle. Surrounded by mountains, lakes, rivers, and forests, a look in any direction is awe-inspiring. However, from being in the heart of these natural gems, Seattle residents also have to deal with rain as almost a daily norm for over half the year. Moreover, a common question asked by Seattle newcomers is "Why is the weather so bipolar here?". Indeed, this is the price that most residents pay. We may dress for a sunny day but will quickly regret this when it suddenly begins to rain in the matter of minutes. Even long-time residents have found it difficult to predict the weather as the skies look deceivingly bright.

**Solution**: To aid residents with being able to understand the land they have grown to love, we are designing a weather forecasting system using embedded machine learning to observe features such as temperature, atmospheric pressure, and relative humidity to predict whether it will rain within the next hour or not. Our device will use the Arduino Nano 33 BLE Sense along with a multi-function environmental module to use a quantized ML model for rain forecasting. We will be using historical data collected from the rooftop of the UW Atmospheric Sciences Building to train our model to be able to forecast rain or no rain using the features we described. Finally, we will package our device and provide it with a battery so that it can send its forecasts via Bluetooth.

**Future Steps**: Beyond the scope of this class, we foresee that our work can be continued to be improved upon by adding useful functionalities such as solar power, air quality detection, and fire prediction. Furthermore, with a network of such systems working in conjunction, we foresee that even forecasting should be possible as we can begin to sense the direction of rain phenomena as well as making our data open access for others to use as a low-cost weather station.

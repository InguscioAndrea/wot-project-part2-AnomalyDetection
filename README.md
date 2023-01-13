The identified Machine Learning algorithm is present in this repository. It will be present on the server and will analyze 
the data taken from the cloud database. The data taken from the sensors, via the Android App, will be uploaded by the latter 
to a cloud database (Firebase, owned by Google). Once stored here, they will be downloaded from the server and placed in the 
appropriate repository, in order to make them available to Machine Learning algorithms.

The algorithm described will carry out Anomaly Detection operations, identifying any anomalies in the values of the three 
quantities of interest (muscle mass, hand grip and walking speed). Once identified, if present, these anomalies will forward 
an email directly to the medical personnel in question, specifying the patient, the quantity in question and the measurement 
(with date and time taken) of interest.
 
Two .PNG images containing the linear trend of the analyzed quantity and a scatter graph of the same will be sent as an 
attachment to each email. Each detected anomaly will be highlighted in red, to differentiate it from a normal value 
displayed in blue.

IMPORTANT NOTE: 
The email used as the recipient for forwarding the anomalies is my personal email. 
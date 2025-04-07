
MeshExpress
Facemesh uitdrukkingsherkenning
Functionele eisen
**Auteurs:**
Kevin van Hoeijen (2118888)
Enes Çaliskan (1671261)
William Hak (1659237)
Jaap-Jan Groenedijk (1548148)
Datum: 17/09/24
**Versiebeheer**
| Versie | Wijziging | Datum |
| --- | --- | --- |
| 0.1 | Initiele opzet | 14-09-2024 |
| 0.2 | Toevoeging test eisen, toegevoegd functionele requirements | 23-09-2024 |
| 1.0 | Laatste versie | 06-11-2024 |
|  |  |  |
Inhoudsopgave
[1. Inleiding 4](#_Toc181783753)
[2. Functioneel design 5](#_Toc181783754)
[2.1 IPO diagram 5](#_Toc181783755)
[2.2 Gebruiker interactie 5](#_Toc181783756)
[3. Vereisten 6](#_Toc181783757)
[3.1 Functionele vereisten 6](#_Toc181783758)
[3.2 Technische vereisten 7](#_Toc181783759)
[4. Planning 8](#_Toc181783760)
# Inleiding
Dit document beschrijft de functionele en technische eisen voor de ontwikkeling van MeshExpress, een systeem dat gezichtsuitdrukkingen herkent aan de hand van camerabeelden.
De technische en functionele vereisten zowel de planning worden vastgelegd voor ontwikkeling van de software om te waarborgen dat het project na oplevering aan de gestelde eisen voldoet van de opdrachtgever, Jeroen Veen namens HAN – University of Applied Sciences .
# Functioneel design
Het begint met een divers dataset dat informatie over een gezicht bevat. Deze zal gelabeld worden, en vervolgens wordt deze dataset gebruikt om het machinelearning algoritme te trainen. Wanneer er een geschikt model is gevonden dan is het systeem klaar voor gebruik.
Er zal een camera zijn die foto’s maakt, welke worden doorgestuurd naar de google face mesh die er 468 locatie punten uit extraheert. Deze punten kunnen gebruikt worden om locaties te herkennen van bijvoorbeeld mondhoeken, wenkbrauwen en dergelijke. Deze geëxtraheerde informatie wordt doorgestuurd naar het machinelearning model die het zal classificeren tot een bepaalde emotie.
## IPO diagram
![A diagram of a software system
Description automatically generated](data:image/png;base64...)
## 2.2 Gebruiker interactie
De gebruiker van de software heeft geen interactie door de meegeleverde API.
# Vereisten
Aan het project zijn de volgende vereisten afgesproken.
## 3.1 Functionele vereisten
| NR | Prioriteit | Requirement |
| --- | --- | --- |
| 1 | Must | Het systeem verwerkt een live videostream. |
| 2 | Must | Het systeem geeft het Face Mesh weer op het gezicht dat het analyseert. |
| 3 | Must | Het systeem kan wenkbrauwen en mond verwerken om gezichtsuitdrukking te herkennen. |
| 4.1 | Must | Het systeem herkent fronsende gezichten |
| 4.2 | Must | Het systeem herkent glimlachende gezichten |
| 4.3 | Could | Het systeem herkent een grote glimlach met zichtbare tanden |
| 4.4 | Could | Het systeem herkent neutrale gezichten |
| 4.5 | Could | Het systeem maakt onderscheid tussen geen herkenning en een neutrale blik |
| 5 | Must | Het systeem kan veranderende gezichtsuitdrukkingen achter elkaar in 1 video kunnen detecteren en volgen. |
| 6 | Must | Het systeem is in staat om emotie detectie resultaten in een bestand te loggen. |
| 7 | Could | Het systeem detecteert wanneer de videostream niet een gezicht van voren bekijkt. |
| 8 | Must | Het systeem kan veranderende gezichtsuitdrukkingen achter elkaar in 1 video kunnen detecteren en volgen. |
| 9 | Must | Het systeem kan automatisch de juiste gezichtsuitdrukking classificeren. |
| 10 | Must | Het systeem heeft een API-interface. |
| 11 | Must | Het systeem kan in de live feed visuele feedback geven over de gedetecteerde gezichtsuitdrukkingen. |
| 12 | Must | Het systeem kan het zekerheidspercentage weergeven over de classificatie |
| 13 | Should | Het systeem kan onderscheid kunnen maken tussen geen herkenning en een neutrale gezichtsuitdrukking |
| 15 | Could | Er is een ROS-2 node |
| 16 | Won’t | Het systeem kan de mogelijkheid bieden om videomateriaal op te slaan met gezichtsherkenningsannotaties. |
## 3.2 Technische vereisten
| NR | Prioriteit | Requirement |
| --- | --- | --- |
| 1 | Must | Het systeem werkt op een generieke laptop |
| 2 | Must | Het systeem werkt op een generieke laptop |
| 3 | Must | Het systeem werkt op ubuntu |
| 4 | Should | Het systeem kan gebruik maken van hardware versnelling d.m.v. een tpu |
| 5 | Must | Het systeem werkt op OUMAX Mini PC Intel N100 16GB 500GB Max N95 8GB 256GB hardware |
| 6 | Must | Het systeem werkt tot 2 meter afstand van de camera |
| 7 | Must | Het systeem is geïmplementeerd in Python |
| 8 | Must | Het systeem blijft werken wanneer een gezicht onder een hoek van 45 graden voor de camera is verdraaid. |
| 9 | Must | Het systeem werkt onder gediffuseerd led verlichting |
| 10 | Must | De vertraging tussen verandering van beeld en bepalen van emotie is niet meer dan 1000ms. |
| 11 | Must | De vertraging tussen opname van emotie en visuele weergave van emotie score is niet meer dan 1000 ms. |
| 12 | Should | Het systeem gebruikt tensorflow light voor microcontroller implementaties |
| 13 | Should | Het programma moet gebruik kunnen maken van hardware versnelling d.m.v een tpu |
| 14 | Could | Het systeem werkt verschillende input beeldresoluties (training & live feed). |
| 15 | Could | Implementatie is in C++ |
# Planning
Fase 1
![A screenshot of a spreadsheet
Description automatically generated](data:image/png;base64...)
Fase 2
![A graph on a white sheet
Description automatically generated](data:image/png;base64...)

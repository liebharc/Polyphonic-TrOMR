# Dataset

This data set consists of public domain pieces, photographed and splitted into individual staff images.

This data set is only used to calculate the symbol error rate (SER). For the SER calculation we want to use a new dataset and not just the validation data
as both PrIMuS and `convert_grandstaff` add artificial distortions to the images and during training the network will likely optimize to read images which have
been distorted in exactly this way. That however doesn't mean in generalizes well, so this data set uses real camera data.

A downside of this data set is that it doesn't contain grandstaff data, that's simply because `music_xml` hasn't yet code to convert grandstaff MusicXml into two semantic parts.

## Source list

https://musescore.com/user/35814389/scores/6428972
http://musescore.com/score/2872906
https://musescore.com/user/13502736/scores/6259652

Open an issue if one of the pieces must not be in this data set. We checked that the license is creative commons.

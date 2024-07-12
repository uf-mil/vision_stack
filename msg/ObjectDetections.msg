Header header

# Information on the time and instance of the set of detections that were made
float64 iteration
uint64 utcTime	# The current UTC time. The year is given as a signed byte year offset from the year 2000. E.g. 2013 as 13.
                # Fields:       year    month    day    hour    min    sec    ms
                # Byte offset:  0       1        2      3       4      5      6|7


# Holds an array of the detections made at a certain instant
ObjectDetection[] detections
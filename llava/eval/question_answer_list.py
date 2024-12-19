QUESTION_ALL = ["Segment the entire image and classify each category separately.",
                "Please perform segmentation on this image and highlight all identifiable elements.",
                "Perform segmentation on this image and label all detected categories.",
                "Please identify and segment all categories present in the image.",
                "Segment the image and label all categories detected.",
                "Could you segment the image and label each identifiable category?",
                "Segment the image to identify and label all visible categories.",
                "Segment and classify all elements in the image.",
                "Identify and segment all categories visible in the image.",
                "Can you segment and label the image?",
                "Might you segment this image?",
                "Can you perform segmentation on this image?",
                "Could you please segment this image?"
             ]

ANSWER_ALL = ["Sure, here is the segmented image with each category classified separately:",
              "Sure, here’s the segmented image showing all visible categories:",
              "The image is segmented and annotated with each category:",
              "The image segmentation is complete, with all categories marked:",
              "Sure, the segmentation mask is:",
              "Sure, the segmented image is:",
              "Certainly, the segmented map is:",
              "Certainly, here is the segmentation mask:",
              "Certainly, here is the segmented output:",
              "Sure, here is the segmentation map:",
              "The segmentation mask is shown below:"
]

# **************************************************************************************

QUESTION_PARTIAL = ["Please segment only the [class_name] in the image.",
                    "Can you segment the [class_name] in the image?",
                    "Where is the [class_name] in this picture? Please respond with segmentation mask.",
                    "Where is '[class_name]' in this image? Please output segmentation mask.",
                    "Could you provide the segmentation mask for '[class_name]' in this image?",
                    "Please segment the image and highlight '[class_name]'."
                     ]


# QUESTION_PARTIAL = ["Can you segment the [class_name] in the image?"
#                     ]


ANSWER_PARTIAL = ["Sure, here is the segmentation mask for '[class_name]':",
                  "Here is the segmentation map focusing on the [class_name]:",
                  "Here is the segmentation mask highlighting the [class_name]:",
                  "The segmentation map for '[class_name]' is:",
                  "The segmentation mask for '[class_name]' is shown below:",
                  "Sure, Here's the segmentation of the [class_name]:",
                  "Sure, the segmented output for '[class_name]' is:",
                  "Certainly, the segmentation map for '[class_name]' is:",
                  "Certainly, here is the segmentation mask for '[class_name]':",
                  "The segmentation mask for '[class_name]' is shown below:"
]

# **************************************************************************************

QUESTION_CONDITION = ["Please segment the image based on the category: [class_name].",
                      "Segment the image according to the specified category: [class_name].",
                      "Segment the image while focusing on the category: [class_name].",
                      "Please provide a segmentation map for the category: [class_name].",
                      "Segment the image with emphasis on the class: [class_name].",
                      "Please segment the image, focusing on the candidate category: [class_name].",
                      "Could you segment the image, considering the indicated class: [class_name]?"
                     ]

ANSWER_CONDITION = ["Sure, here is the segmentation based on the category '[class_name]':",
                    "The image has been segmented according to the category '[class_name]':",
                    "Certainly, here is the segmentation map for the category '[class_name]':",
                    "The image is segmented with emphasis on the class '[class_name]':",
                    "Here is the segmented image focusing on the candidate category '[class_name]':",
                    "The image has been segmented with the category '[class_name]' in mind:",
                    "Sure, the segmentation mask is:",
                    "Sure, the segmented image is:",
                    "Certainly, the segmented map is:",
                    "Certainly, here is the segmentation mask:",
                    "Certainly, here is the segmented output:",
                    "Sure, here is the segmentation map:",
                    "The segmentation mask is shown below:"
]

# **************************************************************************************

QUESTION_OBJECT_ALL = ["Could you please segment the [object] in detail?",
                       "Can you segment the [object] in this image into its parts?",
                       "Segment the [object] and label each part.",
                       "Could you segment the [object] in this image and classify its parts separately?",
                       "Segment the [object] in this image to show all its parts.",
                       "Please provide a detailed segmentation of the [object].",
                       "Segment the [object] in detail and label each identifiable part.",
                       "Please identify and segment the parts of the [object] in this picture."
]

ANSWER_OBJECT_ALL = ["Sure, the segmented map of the [object] is:",
                     "Certainly, here is the detailed segmentation mask of the [object]:",
                     "Here is the segmented image of the [object] with each part classified separately:",
                     "The [object] is segmented and annotated with each part labeled:",
                     "Sure, the detailed segmented image of the [object] is:",
                     "Sure, the segmentation of the [object] into its parts is:"
]

# **************************************************************************************

QUESTION_OBJECT_PART = ["Please segment the [part] of the [object] in this image.",
                        "Can you segment the [part] of the [object]?",
                        "Could you please identify and segment the [part] of the [object] in this image?",
                        "Focus on the [part] of the [object] in this image and segment it, please.",
                        "Please extract the [part] of the [object] from this image and segment it.",
                        "I need the [part] of the [object] segmented in this picture.",
                        "Identify and segment the [part] of the [object] in the given image.",
                        "Segment out the [part] of the [object] shown in this image."
]

ANSWER_OBJECT_PART = ["Sure, here is the segmentation mask for the [part] of the [object]:",
                      "Here is the segmentation map focusing on the [part] of the [object]:",
                      "Here is the segmentation mask highlighting the [part] of the [object]:",
                      "The segmentation map for the [part] of the [object] is:",
                      "Here's the segmentation of the [part] of the [object]:",
                      "Sure, the segmented output for the [part] of the [object] is:",
                      "Certainly, the segmentation map for the [part] of the [object] is:",
                      "Certainly, here is the segmentation mask for the [part] of the [object]:",
                      "Sure, here is the segmentation map for the [part] of the [object]:",
                      "The segmentation mask for the [part] of the [object] is shown below:"
]

# QUESTION_PART_ALL = ["Segment the entire image in detail and classify each part separately.",
#                      "Can you perform detailed segmentation on this image?",
#                      "Could you please segment this image in detail?",
#                      "Might you segment this image in detail?",
#                      "Identify and segment all categories and parts visible in the image.",
#                      "Segment and classify all parts and elements in the image.",
#                      "Please identify and segment all categories and parts present in the image.",
#                      "Segment the image and label all parts detected.",
#                      "Could you segment the image and label each identifiable part?",
#                      "Segment the image to identify and label all visible parts.",
#                      "Segment and classify all parts in the image.",
#                      "Can you segment and label the image parts?",
#                      "Could you please segment this image parts?"
# ]
# ANSWER_PART_ALL = [
#     "Sure, the detailed segmentation mask is:",
#     "The segmented image in detail is:",
#     "Certainly, the detailed segmented map is:",
#     "The detailed segmentation mask is shown below:",
#     "Sure, here’s the segmented image showing all visible parts and categories:",
#     "Here is the segmented image with each part classified separately:",
#     "The image segmentation is complete, with all parts and categories marked:"
# ]

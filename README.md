# FaceRecognitionUsingStreamLit
##Objective: 
Smart attendance system removes the cumbersome task of marking the attendance of the attendees by calling out everyone's name. 

##Problem Statement: 
In today's scenario, taking attendance of students is a headache for the teacher or the CR. Especially in colleges, when there are approx 100 students in each section. Along with this making, a proper record of the attendance days is another irritating task.

##Proposed Solution: 
With our proposed system, there will be no need to fill in the details of every attendee every time. The host/teacher can share the link, opening which the attendee just captures his/her image using the link or can upload his/her picture, the rest work is on us. It overcomes the problem of proxy if we use the live image capturing feature. 

##Algorithm Used: 
Our system is a face recognition system that uses Haar Cascade to detect the face from the image, FACENET to create the embedding of the face, and SVC for classification. We have integrated our system with a Google sheet for maintaining proper attendance i.e. student-wise as well as day-wise. Further, we have used Streamlit and GitHub to design the user-friendly UI for accessing the model and hosting.

##Future Prospective: 
We are looking forward to integrating it with voice assistants so as to make it completely contactless attendance marking system.

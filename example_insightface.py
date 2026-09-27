from insightface_face_similarity import face_similarity_percent

percent = face_similarity_percent("a.jpg", "b.jpg")
if percent is None:
    print("未检测到人脸")
else:
    print(f"{percent:.2f}%")

import api from "./api";

export async function predictCharacter(imageFile) {

    const formData = new FormData();

    formData.append("file", imageFile);

    const response = await api.post(
        "/predict",
        formData,
        {
            headers: {
                "Content-Type": "multipart/form-data"
            }
        }
    );

    return response.data;
}
import api from "./api";

export async function saveFeedback(feedback) {

    const response = await api.post(
        "/feedback",
        feedback
    );

    return response.data;
}
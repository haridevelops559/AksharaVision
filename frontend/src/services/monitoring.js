import api from "./api";

export async function getMonitoring() {

    const response = await api.get(
        "/monitoring"
    );

    return response.data;
}
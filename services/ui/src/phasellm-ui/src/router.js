import { createRouter, createWebHistory } from "vue-router";
import PhaseAISandbox from "./components/PhaseAISandbox.vue";
import PhaseAISimpleLLM from "./components/PhaseAISimpleLLM.vue";
import PhaseAINewsLLM from "./components/PhaseAINewsLLM.vue";

export const router = createRouter({
    history: createWebHistory(),
    routes: [
        { path: "/", component: PhaseAISandbox },
        { path: "/simplellm", component: PhaseAISimpleLLM },
        { path: "/newsllm", component: PhaseAINewsLLM },
    ]
});
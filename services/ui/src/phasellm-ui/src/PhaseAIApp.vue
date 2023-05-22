

<template>
  <!-- <img alt="Vue logo" src="./assets/logo.png"> -->
  <Toast position="bottom-left" />
  <div class="card">
    <Card>
        <template #title>Phase AI Evaluation Platform</template>
        <template #content>
            <p>
                Add (i) a Phase AI elevator pitch and (ii) a description of this app...
            </p>
        </template>
    </Card>
  </div>
  <div class="card">
      <TabMenu :model="mainTabItems" />
      <router-view />
  </div>
</template>

<script>
import axios from "axios";

export default {
  name: 'PhaseAIApp',
  data() {
    return { 
      phasellmPingResponse: "",
      mainTabItems: [
        {
            icon: 'pi pi-fw pi-home',
            label: 'Home',
            to: '/'
        },
        {
            icon: 'pi pi-fw pi-calendar',
            label: 'Simple LLM',
            to: '/simplellm'
        },
        {
            icon: 'pi pi-fw pi-pencil',
            label: 'News LLM',
            to: '/newsllm'
        },
      ]
    }
  },
  async created() {
    try {
      const response = await axios.get(`http://127.0.0.1:5000/ping`);
      console.log(`PhaseAIApp component created. PhaseAI initial ping: ${response}`);
      this.phasellmPingResponse = response.data;

      this.$toast.add(
        {
          severity: 'info',
          summary: 'Info',
          detail: this.phasellmPingResponse,
          life: 4000 
        }
      );
    } catch (error) {
      console.log(error);
    }
  },
}
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
}
</style>

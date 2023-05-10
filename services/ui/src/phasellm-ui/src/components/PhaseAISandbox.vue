<template>

<div class="tabmenudemo-content">
  <div class="card flex justify-content-center">
    A sandbox for LLMs
  </div>
  <div class="card flex justify-content-center">
      <Textarea v-model="quickPromptInput" type="text" placeholder="Type a quick prompt here..." autoResize rows="10" cols="60" />
  </div>
  <div class="card flex justify-content-center">
      <Button @click="promptPhaseAI()" label="Ask LLM" />
  </div>

  <Divider />
  <ProgressBar mode="indeterminate" style="height: 4px" v-if="awaitingPhaseLLM"></ProgressBar>
  
  <div class="card" v-if="phasellmResponse.response">
    <li v-for="choice in phasellmResponse.response.choices" v-bind:key='choice.id'>
      <Card>
        <template #content>
            <div>{{ choice.text }}</div>
        </template>
      </Card>
    </li>
  </div>
</div>

</template>
  
<script>
import axios from "axios";
  
export default {
    name: 'PhaseAIInspector',
    props: {
      welcomeMsg: String
    },
    data() {
      return {
        quickPromptInput: `
Task:
Give five examples of how generative AI can be used.

Format:
A list of generative AI use cases.
        `,
        phasellmResponse: String,
        awaitingPhaseLLM: false
      }
    },
    methods: {
      async promptPhaseAI() {
        this.awaitingPhaseLLM = true;

        try {
          const response = await axios.post(`http://127.0.0.1:5000/prompt`, {
            headers: {
              'Access-Control-Allow-Origin': '*',
              'Content-Type': 'application/json'
            },
            text: this.quickPromptInput,
          });
          console.log(`Test prompt: ${response}`);
          console.log(response);
          this.phasellmResponse = response.data;
        } catch (error) {
          console.log(error);
        } finally {
          this.awaitingPhaseLLM = false;
        }
      },
    },
  }
  </script>
  
<style>
li { list-style-type: none; }
</style>

  
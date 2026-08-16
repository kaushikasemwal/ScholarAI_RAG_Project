pipeline {
    agent any

    environment {
        ACR_NAME = 'scholarairegistry'
        ACR_LOGIN_SERVER = 'scholarairegistry.azurecr.io'
        IMAGE_NAME = 'scholarai-backend'
        RESOURCE_GROUP = 'ScholarAI-RG'
        CONTAINER_APP_NAME = 'scholarai-backend'
        AZURE_SUBSCRIPTION_ID = credentials('AZURE_SUBSCRIPTION_ID')
        AZURE_TENANT_ID = credentials('AZURE_TENANT_ID')
        AZURE_CLIENT_ID = credentials('AZURE_CLIENT_ID')
        AZURE_CLIENT_SECRET = credentials('AZURE_CLIENT_SECRET')
        AES_KEY = credentials('AES_KEY')
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Test') {
            steps {
                sh 'pip install pytest httpx fastapi --quiet'
                sh 'if [ -d tests ]; then pytest tests/ -v --tb=short; else echo "No tests folder, skipping"; fi'
            }
        }

        stage('Docker Build') {
            steps {
                sh "docker build -t ${ACR_LOGIN_SERVER}/${IMAGE_NAME}:latest ."
            }
        }

        stage('Docker Push') {
            steps {
                sh "az login --service-principal -u ${AZURE_CLIENT_ID} -p ${AZURE_CLIENT_SECRET} --tenant ${AZURE_TENANT_ID}"
                sh "az acr login --name ${ACR_NAME}"
                sh "docker push ${ACR_LOGIN_SERVER}/${IMAGE_NAME}:latest"
            }
        }

        stage('Deploy to Azure') {
            steps {
                sh "az containerapp update --name ${CONTAINER_APP_NAME} --resource-group ${RESOURCE_GROUP} --image ${ACR_LOGIN_SERVER}/${IMAGE_NAME}:latest"
            }
        }

        stage('Smoke Test') {
            steps {
                sh 'curl -f https://scholarai-backend.salmonforest-301059c3.centralindia.azurecontainerapps.io/'
            }
        }
    }

    post {
        success {
            echo 'ScholarAI deployed successfully!'
        }
        failure {
            echo 'Pipeline failed. Check logs above.'
        }
    }
}
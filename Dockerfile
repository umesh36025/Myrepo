# Use lightweight Node image
FROM node:18-alpine

# Install Python (needed for your script)
RUN apk add --no-cache python3 py3-pip \
 && ln -sf python3 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy package files first
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy all project files
COPY . .

# Create required folders (safety)
RUN mkdir -p data/uploads pipeline kml_creation

# Expose default port
EXPOSE 3000

# Set production mode
ENV NODE_ENV=production

# Start app
CMD ["npm", "start"]

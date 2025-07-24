# Build stage
FROM node:20-alpine AS build
WORKDIR /app
RUN test -f .env.production || echo -e "VITE_BACKEND_URL=http://127.0.0.1:5001" > .env.production
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

# Production stage
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"] 
import 'dotenv/config';
import http from 'http';
import express, { type NextFunction, type Request, type Response } from 'express';
import cors from 'cors';
import { logger } from './utils/logger';
import { initDB } from './db/init';
import { setupWebSocket } from './ws/audioHandler';
import healthRouter from './routes/health';
import sessionsRouter from './routes/sessions';

const PORT = Number(process.env.PORT) || 4000;

try {
  initDB();
} catch (err) {
  logger.error(err, 'Failed to initialize database');
  process.exit(1);
}

const app = express();
app.use(cors());
app.use(express.json());

app.use('/health', healthRouter);
app.use('/sessions', sessionsRouter);

app.use((err: unknown, _req: Request, res: Response, _next: NextFunction) => {
  logger.error(err);
  res.status(500).json({ error: 'Internal server error' });
});

const server = http.createServer(app);
setupWebSocket(server);

server.listen(PORT, () => {
  logger.info(`VoxGuard backend running on port ${PORT}`);
});

## Run Alembic migrations

### Configurations

```bash
cp alembic.ini.example alembic.ini
```

- update the `alembic.ini` with your database credentials (`sqlalchemy.url`).

### (Optional) Create a new migration

```bash
alembic revision --autogenerate -m "your message"
```

### Apply migrations

```bash
alembic upgrade head
```

### Downgrade migrations

```bash
alembic downgrade -1
```

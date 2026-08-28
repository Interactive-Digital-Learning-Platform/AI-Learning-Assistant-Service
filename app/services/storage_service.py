import aioboto3

from app.core.config import settings


class StorageService:
    def __init__(self):
        self._session = aioboto3.Session()
        self.bucket = settings.S3_BUCKET


    def _client(self):
        return self._session.client(
            "s3",
            endpoint_url=settings.S3_ENDPOINT_URL,
            aws_access_key_id=settings.S3_ACCESS_KEY,
            aws_secret_access_key=settings.S3_SECRET_KEY,
            region_name=settings.S3_REGION
        )

    async def upload(self, key: str, data: bytes, content_type: str) -> None:
        async with self._client() as client:
            await client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=data,
                ContentType=content_type
            )

    async def download(self, key: str) -> bytes:
        async with self._client() as client:
            response = await client.get_object(Bucket=self.bucket, Key=key)
            return await response["Body"].read()


    async def delete(self, key: str) -> None:
        async with self._client() as client:
            await client.delete_object(Bucket=self.bucket, Key=key)


    async def get_preview_url(self, key: str, expires_in: int | None = None) -> str:
        expires_in = expires_in or settings.ATTACHMENT_PREVIEW_URL_EXPIRES_SECONDS
        async with self._client() as client:
            return await client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=expires_in
            )
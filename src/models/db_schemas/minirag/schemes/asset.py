from .minirag_base import SQLAlchemyBase
from sqlalchemy import Column,Integer,String,DateTime,func,ForeignKey
from sqlalchemy.dialects.postgresql import UUID,JSONB
from sqlalchemy.orm import relationship
from sqlalchemy import Index
import uuid


class Asset(SQLAlchemyBase):
    __tablename__ = "assets"

    asset_id = Column(Integer, primary_key=True, autoincrement=True)
    asset_uuid = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True, nullable=False)

    asset_type = Column(String, nullable=False)  # e.g., "image", "document"
    asset_name = Column(String, nullable=False)
    asset_size = Column(Integer, nullable=False)  # Size in bytes
       
    asset_config= Column(JSONB, nullable=True)  # JSONB column for flexible configuration

    asset_project_id = Column(Integer, ForeignKey("projects.project_id"), nullable=False)  # Foreign key to Project table
    project = relationship("Project", back_populates="assets")
    chunks = relationship("DataChunk", back_populates="asset")

    __table_args__ = (
        Index('ix_asset_project_id', asset_project_id),
        Index('ix_asset_type', asset_type),
    )

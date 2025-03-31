import logging
import time
from typing import Optional

from src.model.knowledge_model import KnowledgeModel, Knowledge, KnowledgeForm
from src.service.database.db_connector_service import get_db

log = logging.getLogger(__name__)
log.setLevel("INFO")


class KnowledgeTable:
    @classmethod
    def insert_new_knowledge(
            cls,
            user_id: str,
            knowledge_id: str,
            form_data: KnowledgeForm
    ) -> Optional[KnowledgeModel]:
        with get_db() as db:
            knowledge = KnowledgeModel(
                **{
                    **form_data.model_dump(),
                    "id": knowledge_id,
                    "user_id": user_id,
                    "created_at": int(time.time()),
                    "updated_at": int(time.time()),
                }
            )

            try:
                result = Knowledge(**knowledge.model_dump())
                db.add(result)
                db.commit()
                db.refresh(result)
                return KnowledgeModel.model_validate(result) if result else None
            except Exception:
                return None

    @classmethod
    def get_knowledge_bases(cls) -> list[KnowledgeModel]:
        with get_db() as db:
            knowledge_bases = []
            for knowledge in (
                    db.query(Knowledge).order_by(Knowledge.updated_at.desc()).all()
            ):
                knowledge_bases.append(
                    KnowledgeModel.model_validate(
                        {
                            **KnowledgeModel.model_validate(knowledge).model_dump()
                        }
                    )
                )
            return knowledge_bases

    def get_knowledge_bases_by_user_id(self, user_id: str) -> list[KnowledgeModel]:
        knowledge_bases = self.get_knowledge_bases()
        return [
            knowledge_base
            for knowledge_base in knowledge_bases
            if knowledge_base.user_id == user_id
        ]

    @classmethod
    def get_knowledge_by_id(cls, id: str) -> Optional[KnowledgeModel]:
        try:
            with get_db() as db:
                knowledge = db.query(Knowledge).filter_by(id=id).first()
                return KnowledgeModel.model_validate(knowledge) if knowledge else None
        except Exception:
            return None

    def update_knowledge_by_id(self, id: str, form_data: KnowledgeForm) -> Optional[KnowledgeModel]:
        try:
            with get_db() as db:
                db.query(Knowledge).filter_by(id=id).update(
                    {
                        **form_data.model_dump(),
                        "updated_at": int(time.time()),
                    }
                )
                db.commit()
                return self.get_knowledge_by_id(id=id)
        except Exception as e:
            log.exception(e)
            return None

    def update_knowledge_data_by_id(self, id: str, data: dict) -> Optional[KnowledgeModel]:
        try:
            with get_db() as db:
                db.query(Knowledge).filter_by(id=id).update(
                    {
                        "data": data,
                        "updated_at": int(time.time()),
                    }
                )
                db.commit()
                return self.get_knowledge_by_id(id=id)
        except Exception as e:
            log.exception(e)
            return None

    @classmethod
    def delete_knowledge_by_id(cls, id: str) -> bool:
        try:
            with get_db() as db:
                db.query(Knowledge).filter_by(id=id).delete()
                db.commit()
                return True
        except Exception:
            return False

    @classmethod
    def delete_all_knowledge(cls) -> bool:
        with get_db() as db:
            try:
                db.query(Knowledge).delete()
                db.commit()
                return True
            except Exception:
                return False


Knowledge_table = KnowledgeTable()

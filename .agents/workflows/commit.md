---
description: 변경된 파일 분석 후 git commit & push 자동화
---
# Git Commit Workflow

/commit 명령 실행 시 아래 순서대로 처리:

1. `git status` 와 `git diff` 실행해서 변경 내용 파악
2. 변경 내용 분석 후 Conventional Commits 형식으로 메시지 자동 생성
   - feat: 새 기능 추가
   - fix: 버그 수정
   - refactor: 리팩토링
   - docs: 문서 수정
   - chore: 기타 변경
3. `git add -A` 실행
4. 생성한 메시지로 `git commit -m "..."` 실행
5. `git push origin master` 실행
6. 결과 한국어로 요약 출력
